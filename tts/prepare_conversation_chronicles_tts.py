#!/usr/bin/env python3
"""Prepare per-episode voice-cloned ConversationChronicles TTS shards.

Flattens every multi-session episode into one chronological dialogue, maps the
two speakers onto user/assistant roles (the first speaker of session one is the
user), assigns one reference voice per (episode, role) from a prompt pool, and
emits workload-balanced per-utterance JSONL shards for ``tts/batch_infer.py``
(``--output-format wav``). A flattened ``episodes.jsonl`` keeps the role-mapped
dialogue plus planned audio paths for the later conv_v2 JSON builder.

The prompt pool is a JSON array of reference voices:
  [{"prompt_id": "...", "prompt_audio": "/abs/ref.wav", "prompt_text": "..."}]
Each prompt audio should be a clean 3-10 second 16 kHz mono recording with an
exact transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

SESSIONS = ("first", "second", "third", "fourth", "fifth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path,
                        help="ConversationChronicles raw train.jsonl")
    parser.add_argument("--prompt-pool", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path,
                        help="Root directory where synthesized WAVs will be written")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261121,
                        help="Deterministic salt for per-episode voice assignment")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def pick_prompt(pool: list[dict], episode_id: str, role: str, seed: int,
                excluded_prompt_id: str | None = None) -> dict:
    """Stable per-(episode, role) voice choice; both roles get distinct voices."""
    candidates = [p for p in pool if p["prompt_id"] != excluded_prompt_id] or pool
    digest = hashlib.sha256(f"{seed}:{episode_id}:{role}".encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def flatten_episode(row: dict) -> tuple[list[dict], dict[str, str]]:
    """Return chronological utterances with roles, plus the speaker→role map."""
    utterances: list[dict] = []
    role_by_speaker: dict[str, str] = {}
    for session_index, session in enumerate(SESSIONS):
        dialogue = row.get(f"{session}_session_dialogue") or []
        speakers = row.get(f"{session}_session_speakers") or []
        if len(dialogue) != len(speakers):
            raise ValueError(f"{row['dataID']}: dialogue/speaker length mismatch "
                             f"in {session} session")
        for text, speaker in zip(dialogue, speakers):
            if speaker not in role_by_speaker:
                if len(role_by_speaker) >= 2:
                    raise ValueError(f"{row['dataID']}: more than two speakers")
                role_by_speaker[speaker] = "user" if not role_by_speaker else "assistant"
            text = str(text).strip()
            if not text:
                continue
            utterances.append({
                "session_index": session_index,
                "session": session,
                "time_interval": (row.get("time_interval") or [None] * 5)[session_index],
                "speaker": speaker,
                "role": role_by_speaker[speaker],
                "text": text,
            })
    return utterances, role_by_speaker


def audio_path(audio_root: Path, episode_id: str, utterance_index: int, role: str) -> Path:
    bucket = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()[:2]
    return (audio_root / bucket / episode_id /
            f"u{utterance_index:04d}.{role}.wav")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    pool = json.loads(args.prompt_pool.read_text(encoding="utf-8"))
    if not pool:
        raise ValueError("The prompt pool is empty")
    for entry in pool:
        for key in ("prompt_id", "prompt_audio", "prompt_text"):
            if not entry.get(key):
                raise ValueError(f"Prompt pool entry missing {key}: {entry}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_rows: list[list[dict]] = [[] for _ in range(args.num_shards)]
    shard_weights = [0] * args.num_shards
    counts: Counter = Counter()

    episodes_temp = Path(f"{args.output_dir / 'episodes.jsonl'}.tmp.{os.getpid()}")
    with args.episodes.open(encoding="utf-8") as source, \
            episodes_temp.open("w", encoding="utf-8") as episodes_output:
        emitted = 0
        for line_index, line in enumerate(source):
            if line_index < args.offset or not line.strip():
                continue
            if args.max_episodes is not None and emitted >= args.max_episodes:
                break
            row = json.loads(line)
            episode_id = str(row["dataID"])
            utterances, role_by_speaker = flatten_episode(row)
            if len(utterances) < 2 or len(role_by_speaker) != 2:
                counts["episodes_skipped"] += 1
                continue
            user_prompt = pick_prompt(pool, episode_id, "user", args.seed)
            assistant_prompt = pick_prompt(pool, episode_id, "assistant", args.seed,
                                           excluded_prompt_id=user_prompt["prompt_id"])
            prompts = {"user": user_prompt, "assistant": assistant_prompt}
            records = []
            for utterance_index, utterance in enumerate(utterances):
                role = utterance["role"]
                prompt = prompts[role]
                output = audio_path(args.audio_root, episode_id, utterance_index, role)
                utterance["output_path"] = str(output.resolve())
                records.append({
                    "id": f"{episode_id}_u{utterance_index:04d}_{role}",
                    "conversation_index": episode_id,
                    "conversation_record_count": len(utterances),
                    "turn_index": utterance_index,
                    "role": role,
                    "target_text": utterance["text"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_audio": prompt["prompt_audio"],
                    "prompt_text": prompt["prompt_text"],
                    "output_path": utterance["output_path"],
                    "session_index": utterance["session_index"],
                })
            weight = sum(len(record["target_text"]) for record in records)
            shard_id = min(range(args.num_shards), key=shard_weights.__getitem__)
            shard_rows[shard_id].extend(records)
            shard_weights[shard_id] += weight
            episodes_output.write(json.dumps({
                "episode_id": episode_id,
                "relationship": row.get("relationship"),
                "time_interval": row.get("time_interval"),
                "summary": row.get("summary"),
                "role_by_speaker": role_by_speaker,
                "voice_prompts": {role: prompt["prompt_id"]
                                  for role, prompt in prompts.items()},
                "utterances": utterances,
            }, ensure_ascii=False) + "\n")
            emitted += 1
            counts["episodes_written"] += 1
            counts["tts_utterances"] += len(records)
            counts["target_characters"] += weight
    os.replace(episodes_temp, args.output_dir / "episodes.jsonl")

    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_summaries = []
    for shard_id, rows in enumerate(shard_rows):
        path = shard_dir / f"shard_{shard_id:04d}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in rows:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        shard_summaries.append({
            "shard_id": shard_id,
            "path": str(path.resolve()),
            "record_count": len(rows),
            "target_character_count": shard_weights[shard_id],
        })
    summary = {
        "schema_version": "conversation_chronicles_cosyvoice3_v1",
        "inputs": {
            "episodes": str(args.episodes.resolve()),
            "prompt_pool": str(args.prompt_pool.resolve()),
        },
        "audio_root": str(args.audio_root.resolve()),
        "seed": args.seed,
        "counts": dict(sorted(counts.items())),
        "shards": shard_summaries,
        "voice_policy": {
            "user": "stable per-episode voice from the prompt pool",
            "assistant": "stable per-episode voice, distinct from the user voice",
            "prompt_duration_seconds": "3-10",
            "output_sample_rate": 16000,
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
