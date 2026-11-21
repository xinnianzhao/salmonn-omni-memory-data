# SALMONN-Omni Memory — Data Pipeline

Data construction for
[salmonn-omni-memory](https://github.com/xinnianzhao/salmonn-omni-memory):
**ultra-long, multi-session, two-speaker conversations with full audio**, used
to train and evaluate the TTT long-term-memory model.

Goal: synthetic conversations that span many sessions and tens of minutes to
hours of speech, where later turns depend on information mentioned sessions
earlier. Each episode keeps two stable voices (one per role), is synthesized to
16 kHz mono audio turn by turn, and is finally organized in the upstream
`conv_v2` layout (`data/README.md` of the model repository documents the
training-entry schema).

Current scope:

1. **ConversationChronicles text data** — pinned download + review tooling.
2. **Batch TTS** — CosyVoice3 batched synthesis of every utterance with
   per-episode voice cloning.

## 1. ConversationChronicles

[ConversationChronicles](https://huggingface.co/datasets/jihyoung/ConversationChronicles)
(CC-BY-4.0): 160k episodes × 5 chronological sessions (≈ 800k sessions, 9.4M
utterances, see `conversation_chronicles/train_statistics.json`). Each episode
has a relationship (Classmates, Co-workers, ...), inter-session time intervals
and per-session summaries — exactly the long-horizon, multi-session structure
the memory model needs.

```bash
# pinned revision + hash-verified download into data/conversation_chronicles/
python conversation_chronicles/download.py --data-root data/conversation_chronicles

# human-review exports (uniform session sample / full-episode previews)
python conversation_chronicles/export_review_sample.py
python conversation_chronicles/export_episode_preview.py
```

## 2. Batch TTS (CosyVoice3)

> **Why CosyVoice3?** The upstream SALMONN-Omni stack calls **CosyVoice2-0.5B**,
> but its public API (`inference_zero_shot` / `inference_cross_lingual`) only
> synthesizes one utterance at a time. Only **CosyVoice3**
> (`Fun-CosyVoice3-0.5B-2512`) ships an HF-compatible LLM checkpoint
> (`...-LLM-HF`), which enables true batched `generate` over the speech-token
> LM plus `speech_tokenizer_v3` prompt encoding and `CosyVoice3_Token2Wav`
> vocoding. `tts/batch_infer.py` implements that batched path (Token2Wav still
> iterates per sample upstream, so only the LLM stage is truly batched).

Step 1 — plan shards. Flattens each episode into one chronological dialogue,
maps the first speaker of session one to **user** and the other to
**assistant**, assigns each role a stable reference voice from a prompt pool,
and emits workload-balanced per-utterance shards plus `episodes.jsonl` (the
role-mapped dialogue with planned audio paths, consumed later by the conv_v2
JSON builder):

```bash
python tts/prepare_conversation_chronicles_tts.py \
  --episodes data/conversation_chronicles/raw/train.jsonl \
  --prompt-pool prompts/prompt_pool.json \
  --audio-root data/cc_audio \
  --output-dir data/cc_tts \
  --num-shards 32
```

`prompt_pool.json` is a JSON array of
`{"prompt_id", "prompt_audio", "prompt_text"}` reference voices (clean 3–10 s
16 kHz mono clips with exact transcripts).

Step 2 — synthesize. One GPU per shard; resumable (utterances whose output WAV
already validates are skipped), atomic 16 kHz mono PCM-16 output at each
record's `output_path`, with a per-shard index and metrics JSON:

```bash
export COSYVOICE_REPO=/path/to/CosyVoice          # repo with runtime/triton_trtllm
export COSYVOICE_LLM=/path/to/Fun-CosyVoice3-0.5B-2512-LLM-HF
export COSYVOICE_T2W=/path/to/Fun-CosyVoice3-0.5B-2512
export SHARD_DIR=data/cc_tts OUTPUT_DIR=data/cc_tts/runs
sbatch --array=0-31 --partition=<gpu-partition> --account=<account> tts/batch_infer.slurm

# or directly:
python tts/batch_infer.py \
  --input-jsonl data/cc_tts/shards/shard_0000.jsonl \
  --output-dir data/cc_tts/runs --output-prefix shard0 --output-format wav \
  --cosyvoice-repo "$COSYVOICE_REPO" \
  --llm-model-dir "$COSYVOICE_LLM" --token2wav-model-dir "$COSYVOICE_T2W" \
  --batch-size 8
```

Failed utterances (empty token streams occasionally happen for short numeric
targets) are listed in the metrics JSON; re-running the same command retries
only the missing ones — vary `--seed` / `--temperature` between retries.

## Roadmap

- `tts/build_conv_v2_json.py`: assemble `episodes.jsonl` + synthesized WAVs
  into the final `conv_v2` training JSON for salmonn-omni-memory.
- Backchannel / barge-in augmentation on synthetic dialogues (the GigaSpeech
  real-overlap recipe lives in the model repository's data contract).

## Layout

```
conversation_chronicles/  pinned download + review-sample exports + statistics
tts/                      shard planning, CosyVoice3 batch engine, Slurm template
data/                     (local only, gitignored) downloads and synthesized audio
```
