#!/usr/bin/env bash
# CosyVoice3 batch TTS over prepared shards. Select cluster/partition/account
# with sbatch flags; submit one array element per shard, e.g.
#   sbatch --array=0-31 --partition=<gpu> --account=<acct> tts/batch_infer.slurm
#SBATCH --job-name=cc-batch-tts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Required environment:
#   COSYVOICE_REPO   checkout of CosyVoice (cosyvoice/, third_party/Matcha-TTS,
#                    runtime/triton_trtllm)
#   COSYVOICE_LLM    Fun-CosyVoice3-0.5B-*-LLM-HF directory
#   COSYVOICE_T2W    Fun-CosyVoice3-0.5B-* token2wav directory
#   SHARD_DIR        output of tts/prepare_conversation_chronicles_tts.py
#   OUTPUT_DIR       where per-shard indexes and metrics are written
: "${COSYVOICE_REPO:?}" "${COSYVOICE_LLM:?}" "${COSYVOICE_T2W:?}"
: "${SHARD_DIR:?}" "${OUTPUT_DIR:?}"
PYTHON="${TTS_PYTHON:-python}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_DIR}" logs

printf -v SHARD "%s/shards/shard_%04d.jsonl" "${SHARD_DIR}" "${TASK_ID}"
"${PYTHON}" tts/batch_infer.py \
  --input-jsonl "${SHARD}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "cc_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${TASK_ID}" \
  --output-format wav \
  --cosyvoice-repo "${COSYVOICE_REPO}" \
  --llm-model-dir "${COSYVOICE_LLM}" \
  --token2wav-model-dir "${COSYVOICE_T2W}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --token2wav-group-size "${TOKEN2WAV_GROUP_SIZE:-1}"
