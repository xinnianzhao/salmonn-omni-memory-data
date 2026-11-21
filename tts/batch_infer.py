#!/usr/bin/env python3
"""Offline CosyVoice3 inference with true LLM batching.

Output is either tar-sharded WebDataset-style archives (24 kHz) or direct
16 kHz mono WAV files at each record's ``output_path`` (resumable: records
whose outputs already validate are skipped).

The upstream CosyVoice3 Token2Wav implementation currently iterates over samples
inside Flow and HiFT. This script therefore does not claim Token2Wav GPU batching;
it caches the fixed prompt and groups calls while preserving upstream behavior.

Note: only CosyVoice3 ships the HF-compatible LLM checkpoint
(Fun-CosyVoice3-0.5B-*-LLM-HF) that makes this batched ``generate`` path
possible; the public CosyVoice2 API only exposes per-utterance inference.
"""

import argparse
import io
import json
import os
import random
import re
import sys
import tarfile
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer


SPEECH_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")
SAMPLE_RATE = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--cosyvoice-repo", type=Path, required=True,
                        help="Path to a CosyVoice checkout providing cosyvoice/, "
                             "third_party/Matcha-TTS and runtime/triton_trtllm")
    parser.add_argument("--output-format", choices=("tar", "wav"), default="tar",
                        help="tar: sharded archives at 24 kHz; wav: direct 16 kHz "
                             "mono files at each record's output_path (resumable)")
    parser.add_argument("--llm-model-dir", type=Path, required=True)
    parser.add_argument("--token2wav-model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sort-buffer-multiplier", type=int, default=8)
    parser.add_argument("--token2wav-group-size", type=int, default=1)
    parser.add_argument("--samples-per-tar", type=int, default=1000)
    parser.add_argument("--target-tar-bytes", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_records(path: Path, offset: int, limit: int | None) -> Iterator[dict]:
    emitted = 0
    with path.open(encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            if line_index < offset or not line.strip():
                continue
            if limit is not None and emitted >= limit:
                return
            record = json.loads(line)
            if not record.get("id") or not record.get("target_text"):
                raise ValueError(f"Invalid record at JSONL line {line_index + 1}")
            emitted += 1
            yield record


def length_bucketed_batches(
    records: Iterable[dict], batch_size: int, buffer_multiplier: int
) -> Iterator[list[dict]]:
    iterator = iter(records)
    buffer_size = batch_size * buffer_multiplier
    while True:
        buffer = list(islice(iterator, buffer_size))
        if not buffer:
            return
        buffer.sort(key=lambda item: len(item["target_text"]))
        for start in range(0, len(buffer), batch_size):
            yield buffer[start : start + batch_size]


def load_prompt(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    audio = audio.float().mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio.squeeze(0).contiguous()


def speech_tokens_to_string(tokens: Iterable[int]) -> str:
    return "".join(f"<|s_{int(token)}|>" for token in tokens)


def safe_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not key:
        raise ValueError(f"Cannot construct tar key from id={value!r}")
    return key


TARGET_WAV_RATE = 16_000


def wav_valid(path: Path) -> bool:
    """A finished output: 16 kHz mono PCM WAV with at least one frame."""
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        info = sf.info(path)
        return (
            info.samplerate == TARGET_WAV_RATE
            and info.channels == 1
            and info.frames > 0
            and info.format == "WAV"
        )
    except (OSError, RuntimeError):
        return False


class DirectWavWriter:
    """Write each utterance as a 16 kHz mono WAV at ``record['output_path']``."""

    def __init__(self, output_dir: Path, prefix: str, samples_per_tar: int, target_tar_bytes: int):
        del samples_per_tar, target_tar_bytes
        self.output_dir = Path(output_dir).resolve()
        self.prefix = prefix
        self.total_samples = 0
        self.total_conversations = 0
        self.oversized_conversations = 0
        self.shard_summaries: list[dict] = []
        self.index_rows: list[dict] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, path: Path, waveform: np.ndarray) -> None:
        tensor = torch.as_tensor(waveform, dtype=torch.float32).flatten()
        if tensor.numel() == 0:
            raise ValueError(f"Empty CosyVoice waveform for {path}")
        resampled = torchaudio.functional.resample(tensor, SAMPLE_RATE, TARGET_WAV_RATE)
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{resolved}.tmp.{os.getpid()}")
        try:
            sf.write(temporary, resampled.cpu().numpy(), TARGET_WAV_RATE, format="WAV", subtype="PCM_16")
            os.replace(temporary, resolved)
        finally:
            temporary.unlink(missing_ok=True)
        if not wav_valid(resolved):
            raise RuntimeError(f"Invalid synthesized WAV after write: {resolved}")

    def add_conversation(self, conversation_items: list[tuple[dict, np.ndarray]]) -> None:
        if not conversation_items:
            return
        for record, waveform in sorted(
            conversation_items, key=lambda item: int(item[0].get("turn_index", 0))
        ):
            path = Path(record["output_path"])
            self._write(path, waveform)
            info = sf.info(path)
            self.index_rows.append(
                {
                    "id": record["id"],
                    "conversation_index": record.get("conversation_index"),
                    "turn_index": record.get("turn_index"),
                    "role": record.get("role"),
                    "path": str(path.resolve()),
                    "sample_rate": info.samplerate,
                    "frames": info.frames,
                    "duration_seconds": info.frames / info.samplerate,
                }
            )
            self.total_samples += 1
        self.total_conversations += 1

    def add(self, record: dict, waveform: np.ndarray) -> None:
        self.add_conversation([(record, waveform)])

    def close(self) -> None:
        index_path = self.output_dir / f"{self.prefix}-index.jsonl"
        temporary = Path(f"{index_path}.tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in self.index_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, index_path)
        self.shard_summaries = [
            {
                "path": str(index_path.resolve()),
                "utterance_count": self.total_samples,
                "conversation_count": self.total_conversations,
            }
        ]


def resumable_wav_records(records, log=print):
    """Skip whole conversations whose output WAVs already validate."""
    from collections import defaultdict

    grouped: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    total = reused = 0
    for record in records:
        if not record.get("output_path"):
            raise ValueError(f"Record {record['id']} needs output_path in wav mode")
        key = str(record.get("conversation_index", record["id"]))
        if key not in grouped:
            order.append(key)
        grouped[key].append(record)
        total += 1
    for key in order:
        group = grouped[key]
        if all(wav_valid(Path(item["output_path"])) for item in group):
            reused += len(group)
            continue
        yield from group
    log(f"resumable_input={total} reused_existing={reused}")


class TarShardWriter:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        samples_per_tar: int,
        target_tar_bytes: int,
    ):
        self.output_dir = output_dir
        self.prefix = prefix
        self.samples_per_tar = samples_per_tar
        self.target_tar_bytes = target_tar_bytes
        self.shard_index = 0
        self.shard_samples = 0
        self.shard_conversations = 0
        self.total_samples = 0
        self.total_conversations = 0
        self.oversized_conversations = 0
        self.shard_summaries: list[dict] = []
        self.archive: tarfile.TarFile | None = None
        self.incomplete_path: Path | None = None
        self.final_path: Path | None = None
        output_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        self.final_path = self.output_dir / f"{self.prefix}-{self.shard_index:05d}.tar"
        self.incomplete_path = self.final_path.with_suffix(".tar.incomplete")
        if self.final_path.exists() or self.incomplete_path.exists():
            raise FileExistsError(f"Refusing to overwrite {self.final_path}")
        self.archive = tarfile.open(self.incomplete_path, mode="w")
        self.shard_samples = 0
        self.shard_conversations = 0

    def _add_bytes(self, name: str, payload: bytes) -> None:
        if self.archive is None:
            raise RuntimeError("tar archive is not open")
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mtime = int(time.time())
        self.archive.addfile(info, io.BytesIO(payload))

    @staticmethod
    def _member_storage_bytes(payload: bytes) -> int:
        return 512 + ((len(payload) + 511) // 512) * 512

    def _should_rotate(self, sample_count: int, storage_bytes: int) -> bool:
        if self.archive is None or self.shard_samples == 0:
            return False
        sample_limit_hit = (
            self.samples_per_tar > 0
            and self.shard_samples + sample_count > self.samples_per_tar
        )
        byte_limit_hit = (
            self.target_tar_bytes > 0
            and self.archive.offset + storage_bytes + 10240 > self.target_tar_bytes
        )
        return sample_limit_hit or byte_limit_hit

    def add_conversation(
        self, conversation_items: list[tuple[dict, np.ndarray]]
    ) -> None:
        if not conversation_items:
            return

        members: list[tuple[str, bytes]] = []
        utterances = []
        for record, waveform in sorted(
            conversation_items, key=lambda item: int(item[0].get("turn_index", 0))
        ):
            key = safe_key(str(record["id"]))
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            metadata = dict(record)
            metadata.update(
                {
                    "audio": f"{key}.wav",
                    "sample_rate": SAMPLE_RATE,
                    "num_samples": int(waveform.shape[-1]),
                    "duration_seconds": float(waveform.shape[-1] / SAMPLE_RATE),
                }
            )
            members.append((f"{key}.wav", wav_buffer.getvalue()))
            members.append(
                (
                    f"{key}.json",
                    json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
                )
            )
            utterances.append(
                {
                    "id": record["id"],
                    "turn_index": record.get("turn_index"),
                    "role": record.get("role"),
                    "audio": f"{key}.wav",
                    "metadata": f"{key}.json",
                }
            )

        first_record = conversation_items[0][0]
        conversation_index = first_record.get("conversation_index")
        if conversation_index is None:
            conversation_key = safe_key(str(first_record["id"]))
        else:
            conversation_key = f"conv_{int(conversation_index):06d}"
        conversation_metadata = {
            "conversation_index": conversation_index,
            "utterance_count": len(utterances),
            "utterances": utterances,
        }
        members.append(
            (
                f"{conversation_key}.conversation.json",
                json.dumps(conversation_metadata, ensure_ascii=False).encode("utf-8"),
            )
        )
        storage_bytes = sum(self._member_storage_bytes(payload) for _, payload in members)

        if self.archive is None:
            self._open()
        elif self._should_rotate(len(conversation_items), storage_bytes):
            self._close_current()
            self.shard_index += 1
            self._open()
        if self.target_tar_bytes > 0 and storage_bytes + 10240 > self.target_tar_bytes:
            self.oversized_conversations += 1
        for name, payload in members:
            self._add_bytes(name, payload)
        self.shard_samples += len(conversation_items)
        self.shard_conversations += 1
        self.total_samples += len(conversation_items)
        self.total_conversations += 1

    def add(self, record: dict, waveform: np.ndarray) -> None:
        self.add_conversation([(record, waveform)])

    def _close_current(self) -> None:
        if self.archive is None:
            return
        self.archive.close()
        self.archive = None
        if self.incomplete_path is None or self.final_path is None:
            raise RuntimeError("tar paths are not initialized")
        os.replace(self.incomplete_path, self.final_path)
        self.shard_summaries.append(
            {
                "path": str(self.final_path.resolve()),
                "bytes": self.final_path.stat().st_size,
                "utterance_count": self.shard_samples,
                "conversation_count": self.shard_conversations,
            }
        )

    def close(self) -> None:
        self._close_current()


class CosyVoice3BatchRuntime:
    def __init__(self, args: argparse.Namespace):
        repo_root = Path(args.cosyvoice_repo).expanduser().resolve()
        runtime_dir = repo_root / "runtime" / "triton_trtllm"
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, str(repo_root / "third_party" / "Matcha-TTS"))
        sys.path.insert(0, str(runtime_dir))

        import s3tokenizer
        from token2wav_cosyvoice3 import CosyVoice3_Token2Wav

        self.args = args
        self.device = torch.device("cuda:0")
        self.prompt_audio_cache: dict[str, torch.Tensor] = {}
        self.seen_prompt_ids: set[str] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_dir)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = AutoModelForCausalLM.from_pretrained(
            args.llm_model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().to(self.device)

        self.s3_tokenizer = s3tokenizer.load_model(
            str(args.token2wav_model_dir / "speech_tokenizer_v3.onnx")
        ).to(self.device).eval()
        self.token2wav = CosyVoice3_Token2Wav(
            model_dir=str(args.token2wav_model_dir),
            enable_trt=False,
            device_id=0,
            streaming=False,
        )

    def _resolve_prompt(self, record: dict) -> tuple[str, str, torch.Tensor]:
        audio_value = record.get("prompt_audio") or self.args.prompt_audio
        text_value = record.get("prompt_text") or self.args.prompt_text
        if not audio_value or not text_value:
            raise ValueError(
                f"Record {record['id']} needs prompt_audio and prompt_text, "
                "either in JSONL or through global CLI arguments"
            )
        audio_path = str(Path(audio_value).resolve())
        if audio_path not in self.prompt_audio_cache:
            self.prompt_audio_cache[audio_path] = load_prompt(Path(audio_path))
        prompt_id = str(record.get("prompt_id") or audio_path)
        self.seen_prompt_ids.add(prompt_id)
        return audio_path, str(text_value), self.prompt_audio_cache[audio_path]

    def _build_input_ids(
        self,
        target_text: str,
        prompt_text: str,
        prompt_speech_token_string: str,
    ) -> torch.Tensor:
        full_text = (
            "You are a helpful assistant.<|endofprompt|>"
            + prompt_text
            + target_text
        )
        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_speech_token_string},
        ]
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).squeeze(0)

    @torch.inference_mode()
    def generate_speech_tokens(
        self, records: list[dict]
    ) -> tuple[list[list[int]], list[torch.Tensor]]:
        import s3tokenizer

        resolved = [self._resolve_prompt(record) for record in records]
        unique_audio: dict[str, torch.Tensor] = {}
        for audio_path, _, audio in resolved:
            unique_audio.setdefault(audio_path, audio)

        unique_paths = list(unique_audio)
        prompt_mels = [
            s3tokenizer.log_mel_spectrogram(unique_audio[path])
            for path in unique_paths
        ]
        padded, lengths = s3tokenizer.padding(prompt_mels)
        codes, code_lengths = self.s3_tokenizer.quantize(
            padded.to(self.device), lengths.to(self.device)
        )
        token_strings = {
            path: speech_tokens_to_string(codes[index, : code_lengths[index].item()])
            for index, path in enumerate(unique_paths)
        }

        sequences = [
            self._build_input_ids(
                record["target_text"], prompt_text, token_strings[audio_path]
            )
            for record, (audio_path, prompt_text, _) in zip(records, resolved)
        ]
        max_length = max(sequence.shape[0] for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, -sequence.shape[0] :] = sequence
            attention_mask[index, -sequence.shape[0] :] = 1

        outputs = self.llm.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            repetition_penalty=1.1,
            top_k=self.args.top_k,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[:, max_length:].cpu()
        result = []
        for sequence in generated:
            decoded = self.tokenizer.decode(sequence, skip_special_tokens=False)
            result.append([int(value) for value in SPEECH_TOKEN_RE.findall(decoded)])
        return result, [audio for _, _, audio in resolved]

    @torch.inference_mode()
    def tokens_to_waveforms(
        self,
        speech_tokens: list[list[int]],
        prompt_audios: list[torch.Tensor],
    ) -> list[np.ndarray]:
        waveforms = self.token2wav(
            speech_tokens,
            prompt_audios,
            [16_000] * len(prompt_audios),
            streaming=False,
        )
        return [waveform.squeeze().float().cpu().numpy() for waveform in waveforms]


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.token2wav_group_size < 1
        or args.samples_per_tar < 0
        or args.target_tar_bytes < 0
    ):
        raise ValueError("batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if bool(args.prompt_audio) != bool(args.prompt_text):
        raise ValueError("Global prompt-audio and prompt-text must be provided together")
    runtime = CosyVoice3BatchRuntime(args)
    writer_class = DirectWavWriter if args.output_format == "wav" else TarShardWriter
    writer = writer_class(
        args.output_dir,
        args.output_prefix,
        args.samples_per_tar,
        args.target_tar_bytes,
    )
    records = load_records(args.input_jsonl, args.offset, args.limit)
    if args.output_format == "wav":
        records = resumable_wav_records(records)
    batches = length_bucketed_batches(
        records, args.batch_size, args.sort_buffer_multiplier
    )

    started = time.time()
    input_count = 0
    failed_ids: list[str] = []
    llm_seconds = 0.0
    token2wav_seconds = 0.0
    pending_conversations: dict[str, dict] = {}
    dropped_conversations: list[int | str] = []

    def record_result(record: dict, waveform: np.ndarray | None) -> None:
        conversation_id = record.get("conversation_index", record["id"])
        key = str(conversation_id)
        expected = int(record.get("conversation_record_count", 1))
        state = pending_conversations.setdefault(
            key, {"conversation_id": conversation_id, "expected": expected, "seen": 0, "items": []}
        )
        if state["expected"] != expected:
            raise ValueError(f"Inconsistent conversation_record_count for {conversation_id}")
        state["seen"] += 1
        if waveform is not None:
            state["items"].append((record, waveform))
        if state["seen"] == state["expected"]:
            if len(state["items"]) == state["expected"]:
                writer.add_conversation(state["items"])
            else:
                dropped_conversations.append(state["conversation_id"])
            del pending_conversations[key]
    try:
        for batch_index, batch in enumerate(batches):
            input_count += len(batch)
            llm_started = time.time()
            token_lists, prompt_audios = runtime.generate_speech_tokens(batch)
            llm_seconds += time.time() - llm_started

            valid = [
                (record, tokens, prompt_audio)
                for record, tokens, prompt_audio in zip(
                    batch, token_lists, prompt_audios
                )
                if tokens
            ]
            for record, tokens in zip(batch, token_lists):
                if not tokens:
                    failed_ids.append(str(record["id"]))
                    record_result(record, None)

            for start in range(0, len(valid), args.token2wav_group_size):
                group = valid[start : start + args.token2wav_group_size]
                token2wav_started = time.time()
                waveforms = runtime.tokens_to_waveforms(
                    [tokens for _, tokens, _ in group],
                    [prompt_audio for _, _, prompt_audio in group],
                )
                token2wav_seconds += time.time() - token2wav_started
                for (record, _, _), waveform in zip(group, waveforms):
                    record_result(record, waveform)

            print(
                f"batch={batch_index} input={input_count} written={writer.total_samples} "
                f"failed={len(failed_ids)}",
                flush=True,
            )
        for state in pending_conversations.values():
            dropped_conversations.append(state["conversation_id"])
        pending_conversations.clear()
    finally:
        writer.close()

    metrics = {
        "input_count": input_count,
        "written_count": writer.total_samples,
        "written_conversation_count": writer.total_conversations,
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "dropped_conversation_count": len(dropped_conversations),
        "dropped_conversations": dropped_conversations,
        "llm_seconds": llm_seconds,
        "token2wav_seconds": token2wav_seconds,
        "total_seconds": time.time() - started,
        "llm_batch_size": args.batch_size,
        "token2wav_true_batch": False,
        "token2wav_group_size": args.token2wav_group_size,
        "unique_prompt_count": len(runtime.seen_prompt_ids),
        "target_tar_bytes": args.target_tar_bytes,
        "samples_per_tar": args.samples_per_tar,
        "oversized_conversation_count": writer.oversized_conversations,
        "tar_shards": writer.shard_summaries,
    }
    metrics_path = args.output_dir / f"{args.output_prefix}-metrics.json"
    with metrics_path.open("x", encoding="utf-8") as destination:
        json.dump(metrics, destination, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
