#!/usr/bin/env python3
"""Expand previously reviewed sessions into complete episodes with time gaps.

Select the first N distinct episodes in the existing session review order.
Keep one speaker-to-role mapping across all five sessions of each episode.
Also recount individual utterances over the complete local training split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from download import sha256_file, write_json

SESSION_NAMES = ("first", "second", "third", "fourth", "fifth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/conversation_chronicles"))
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    root = args.data_root.resolve()
    source = root / "raw" / "train.jsonl"
    review_manifest_path = root / "review_sample_100.manifest.json"
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    selected = {}
    for sample in review_manifest["samples"]:
        selected.setdefault(sample["episode_id"], sample)
        if len(selected) == args.count:
            break
    if len(selected) < args.count:
        raise ValueError("Too few distinct episodes in the existing review sample")
    source_hash = hashlib.sha256()
    episodes = sessions = turns = 0
    selected_rows = {}
    print("Expanding review episodes and recounting the complete training split...", flush=True)
    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            source_hash.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            episodes += 1
            for name in SESSION_NAMES:
                dialogue = row[f"{name}_session_dialogue"]
                if not isinstance(dialogue, list):
                    raise ValueError(f"Invalid dialogue at line {line_number}, {name}")
                sessions += 1
                turns += len(dialogue)
            if row["dataID"] in selected:
                if row["dataID"] in selected_rows:
                    raise ValueError("Duplicate selected episode ID")
                if line_number != selected[row["dataID"]]["source_line_number"]:
                    raise ValueError("Episode does not match the previous review source")
                selected_rows[row["dataID"]] = row
    if source_hash.hexdigest() != review_manifest["source_sha256"]:
        raise ValueError("Source file differs from the previous review source")
    if len(selected_rows) != args.count:
        raise ValueError("Some selected episodes are missing")
    preview, provenance = [], []
    for episode_id, anchor in selected.items():
        row = selected_rows[episode_id]
        labels = list(dict.fromkeys(
            speaker for name in SESSION_NAMES for speaker in row[f"{name}_session_speakers"]
        ))
        if len(labels) != 2:
            raise ValueError(f"Expected two stable speaker labels in {episode_id}")
        speaker_to_role = dict(zip(labels, ("user", "assistant")))
        output_episode = {
            "episode_id": episode_id,
            "relationship": row["relationship"],
            "speakers": {role: speaker for speaker, role in speaker_to_role.items()},
            "sessions": [],
        }
        for index, name in enumerate(SESSION_NAMES):
            dialogue = row[f"{name}_session_dialogue"]
            speakers = row[f"{name}_session_speakers"]
            if len(dialogue) != len(speakers) or not all(isinstance(text, str) for text in dialogue):
                raise ValueError(f"Invalid speaker/text alignment in {episode_id}, {name}")
            output_episode["sessions"].append({
                "session_number": index + 1,
                "time_since_previous_session": row["time_interval"][index],
                "dialogue": [{speaker_to_role[speaker]: text} for speaker, text in zip(speakers, dialogue)],
            })
        preview.append(output_episode)
        provenance.append({
            "preview_index": len(preview) - 1,
            "episode_id": episode_id,
            "source_line_number": anchor["source_line_number"],
            "original_session_review_index": anchor["review_index"],
            "original_review_session_number": anchor["session_number"],
        })
    output = root / f"review_episodes_{args.count}.json"
    write_json(output, preview)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    if reloaded != preview:
        raise ValueError("Serialized preview differs from the intended export")
    statistics = {
        "split": "train", "source_sha256": source_hash.hexdigest(),
        "turn_definition": "One original speaker utterance, not a user-assistant exchange",
        "episodes": episodes, "sessions": sessions, "turns": turns,
        "mean_turns_per_episode": turns / episodes,
        "mean_turns_per_session": turns / sessions,
        "mean_sessions_per_episode": sessions / episodes,
    }
    write_json(root / "train_statistics.json", statistics)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": review_manifest["repo_id"], "revision": review_manifest["revision"],
        "source": str(source), "source_sha256": source_hash.hexdigest(), "split": "train",
        "selection": "First N distinct episodes in the previous random session review order; no content filtering",
        "selection_manifest_sha256": sha256_file(review_manifest_path),
        "speaker_mapping": "First encountered speaker in the episode -> user; other speaker -> assistant; fixed across every session",
        "time_intervals": "Original relative gaps from the previous session, not absolute dates",
        "text_preserved_verbatim": True, "truncated": False,
        "preview_episodes": len(preview), "preview_sessions": sum(len(e["sessions"]) for e in preview),
        "preview_turns": sum(len(s["dialogue"]) for e in preview for s in e["sessions"]),
        "output_sha256": sha256_file(output), "export_script_sha256": sha256_file(Path(__file__)),
        "samples": provenance,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"statistics": statistics, "preview_episodes": len(preview),
                      "preview_turns": manifest["preview_turns"], "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
