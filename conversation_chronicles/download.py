#!/usr/bin/env python3
"""Download and verify the pinned Conversation Chronicles training split.

Requires huggingface_hub; validation/test data are not downloaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

REPO_ID = "jihyoung/ConversationChronicles"
REVISION = "9639b3552594a612ceafa76238a34f0f6eeaa207"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/conversation_chronicles"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / ".cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    files = {item.rfilename: item for item in info.siblings}
    manifest_path = root / "source_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["revision"] != info.sha or previous["repo_id"] != REPO_ID:
            raise ValueError("Use a different data root for a different dataset snapshot")
    manifest = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "snapshot_url": f"https://huggingface.co/datasets/{REPO_ID}/tree/{info.sha}",
        "license": info.card_data.to_dict().get("license"),
        "split": "train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "files": [],
    }
    write_json(manifest_path, manifest)
    for filename in ("README.md", "train.jsonl"):
        entry = files[filename]
        print(f"Downloading {filename}: {entry.size:,} bytes", flush=True)
        path = Path(hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename,
            revision=info.sha, local_dir=root / "raw", token=False,
        ))
        if path.stat().st_size != entry.size:
            raise ValueError(f"File size mismatch: {filename}")
        digest = sha256_file(path)
        expected_hash = entry.lfs.sha256 if entry.lfs else None
        if expected_hash and digest != expected_hash:
            raise ValueError(f"SHA256 mismatch: {filename}")
        if not entry.lfs:
            git_digest = hashlib.sha1(f"blob {entry.size}\0".encode() + path.read_bytes()).hexdigest()
            if git_digest != entry.blob_id:
                raise ValueError(f"Git blob hash mismatch: {filename}")
        manifest["files"].append({
            "path": filename, "size_bytes": entry.size,
            "expected_sha256": expected_hash, "sha256": digest,
            "git_blob_id": entry.blob_id, "verified": True,
        })
        write_json(manifest_path, manifest)
        print(f"Verified {filename}", flush=True)
    manifest["complete"] = True
    manifest["download_finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print("Complete training split downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
