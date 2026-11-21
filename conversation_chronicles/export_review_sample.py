#!/usr/bin/env python3
"""Uniformly sample complete training sessions into a text-only review JSON.

Each top-level item is one session: a list of {"user": text} / {"assistant":
text} messages. Actual speaker labels determine roles, so consecutive turns by
one speaker and odd-length sessions are preserved. Provenance is a sidecar.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from download import sha256_file, write_json

SESSION_NAMES = ("first", "second", "third", "fourth", "fifth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/conversation_chronicles"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("sample-size must be positive")
    root = args.data_root.resolve()
    source = root / "raw" / "train.jsonl"
    output = args.output or root / f"review_sample_{args.sample_size}.json"
    download_manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if not download_manifest["complete"] or download_manifest["split"] != "train":
        raise ValueError("A complete, verified training download is required")
    source_entry = next(item for item in download_manifest["files"] if item["path"] == "train.jsonl")
    rng = random.Random(args.seed)
    reservoir = []
    source_digest = hashlib.sha256()
    episode_count = session_count = source_turns = 0
    print("Sampling all training sessions with reservoir sampling...", flush=True)
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            source_digest.update(raw_line)
            if not raw_line.strip():
                continue
            episode = json.loads(raw_line)
            episode_count += 1
            for session_number, name in enumerate(SESSION_NAMES, start=1):
                dialogue = episode[f"{name}_session_dialogue"]
                if not isinstance(dialogue, list):
                    raise ValueError(f"Expected dialogue list at line {line_number}, {name}")
                source_turns += len(dialogue)
                session_count += 1
                slot = session_count - 1 if session_count <= args.sample_size else rng.randrange(session_count)
                if slot >= args.sample_size:
                    continue
                selected = {
                    "source_line_number": line_number,
                    "flat_session_index": session_count - 1,
                    "episode_id": episode["dataID"],
                    "session_number": session_number,
                    "session_field": f"{name}_session_dialogue",
                    "relationship": episode["relationship"],
                    "time_interval": episode["time_interval"][session_number - 1],
                    "speakers": episode[f"{name}_session_speakers"],
                    "dialogue": dialogue,
                }
                if session_count <= args.sample_size:
                    reservoir.append(selected)
                else:
                    reservoir[slot] = selected
            if episode_count % 40000 == 0:
                print(f"Scanned {episode_count:,} episodes / {session_count:,} sessions", flush=True)
    if source_digest.hexdigest() != source_entry["sha256"]:
        raise ValueError("Training file differs from the verified download")
    if session_count < args.sample_size:
        raise ValueError("Requested sample is larger than the session population")
    rng.shuffle(reservoir)
    review, provenance = [], []
    session_positions = Counter()
    odd_length_sessions = consecutive_speaker_sessions = sample_turns = 0
    for review_index, selected in enumerate(reservoir):
        dialogue, speakers = selected["dialogue"], selected["speakers"]
        speaker_labels = list(dict.fromkeys(speakers))
        if len(dialogue) != len(speakers) or len(speaker_labels) != 2:
            raise ValueError(f"Cannot map speakers faithfully: {selected['episode_id']}, session {selected['session_number']}")
        if not all(isinstance(text, str) for text in dialogue):
            raise ValueError("Sample contains non-string utterances")
        speaker_to_role = dict(zip(speaker_labels, ("user", "assistant")))
        messages = [{speaker_to_role[speaker]: text} for speaker, text in zip(speakers, dialogue)]
        review.append(messages)
        provenance.append({
            "review_index": review_index,
            **{key: value for key, value in selected.items() if key not in ("dialogue", "speakers")},
            "speaker_to_display_role": speaker_to_role,
            "original_speakers": speakers,
            "turns": len(dialogue),
        })
        session_positions[selected["session_number"]] += 1
        sample_turns += len(dialogue)
        odd_length_sessions += len(dialogue) % 2
        consecutive_speaker_sessions += any(a == b for a, b in zip(speakers, speakers[1:]))
    write_json(output, review)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    if len(reloaded) != args.sample_size or len({item["flat_session_index"] for item in reservoir}) != args.sample_size:
        raise ValueError("Incorrect sample size or duplicate source sessions")
    for messages, selected in zip(reloaded, reservoir):
        if [next(iter(message.values())) for message in messages] != selected["dialogue"]:
            raise ValueError("Export changed or truncated dialogue text")
        if not all(len(message) == 1 and set(message) <= {"user", "assistant"} for message in messages):
            raise ValueError("Unexpected metadata in review JSON")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": download_manifest["repo_id"],
        "revision": download_manifest["revision"],
        "split": "train", "source": str(source), "source_sha256": source_digest.hexdigest(),
        "population_episodes": episode_count, "population_sessions": session_count,
        "population_turns": source_turns,
        "sampling": "Uniform without replacement over individual sessions; reservoir algorithm R; shuffled output",
        "seed": args.seed, "filters": [],
        "sample_sessions": len(review), "sample_turns": sample_turns,
        "sample_session_positions": dict(sorted(session_positions.items())),
        "sample_odd_length_sessions": odd_length_sessions,
        "sample_consecutive_speaker_sessions": consecutive_speaker_sessions,
        "role_mapping": "Within each sampled session, first encountered original speaker -> user; other speaker -> assistant. Display labels only; original data are human-human dialogue.",
        "text_preserved_verbatim": True, "truncated": False,
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
        "export_script_sha256": sha256_file(Path(__file__)),
        "samples": provenance,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({
        "population_episodes": episode_count, "population_sessions": session_count,
        "sample_sessions": len(review), "sample_turns": sample_turns,
        "odd_length_sessions": odd_length_sessions,
        "consecutive_speaker_sessions": consecutive_speaker_sessions,
        "output": str(output), "verified": True,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
