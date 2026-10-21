"""Check real subtitle chunk boundaries without changing the recommendation database."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from video_agent.config import Settings
from video_agent.rag import CHUNK_POLICY, _partition, _semantic_distances, _subtitle_pieces
from video_agent.rag import split_subtitles
from video_agent.runtime import build_models
from video_agent.schemas import Video


class CachedEmbedding:
    def __init__(self, model):
        self.model, self.cache = model, {}

    def embed(self, texts):
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = self.model.embed(texts)
        return self.cache[key]


def validate_video(video, embedding, labels, tolerance):
    pieces = _subtitle_pieces(video)
    chunks = split_subtitles(video, embedding)
    distances = _semantic_distances(pieces, embedding)
    threshold = float(np.percentile(list(distances.values()), CHUNK_POLICY["percentile"])) \
        if distances else 0.0
    ends = [int(chunk.id.split(":")[-2]) for chunk in chunks[:-1]]
    cuts = [{"piece_end": end, "seconds": pieces[end - 1].end,
             "distance": distances.get(end),
             "semantic": distances.get(end, 0.0) > threshold + 1e-9} for end in ends]
    checks = []
    for label in labels:
        feasible = False
        for end in range(1, len(pieces)):
            if label["start"] <= pieces[end - 1].end <= label["end"]:
                try:
                    _partition(pieces[:end], {})
                    _partition(pieces[end:], {})
                    feasible = True
                    break
                except ValueError:
                    pass
        nearest = min(cuts, key=lambda c: max(label["start"] - c["seconds"],
                                              c["seconds"] - label["end"], 0)) if cuts else None
        gap = max(label["start"] - nearest["seconds"], nearest["seconds"] - label["end"], 0) \
            if nearest else None
        checks.append({**label, "length_feasible": feasible, "nearest_cut": nearest,
                       "gap_seconds": gap, "hit": feasible and gap is not None and gap <= tolerance})
    lengths = [len(chunk.text) for chunk in chunks]
    return {"bvid": video.bvid, "segments": len(video.segments),
            "windows": len(distances) * 2, "boundary_candidates": len(distances),
            "threshold": threshold,
            "distances": [{"piece_end": end, "seconds": pieces[end - 1].end, "distance": d}
                          for end, d in distances.items()],
            "cuts": cuts, "boundary_checks": checks, "chunk_lengths": lengths,
            "length_passed": all(CHUNK_POLICY["min_chars"] <= n <= CHUNK_POLICY["max_chars"]
                                 for n in lengths),
            "chunks": [chunk.model_dump() for chunk in chunks]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("tests/fixtures/chunk-boundaries.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    models = build_models(settings)
    started = time.monotonic()
    try:
        embedding = CachedEmbedding(models.embedding)
        videos = [validate_video(Video.model_validate_json(
            (args.input_dir / f"{bvid}.json").read_text(encoding="utf-8")), embedding, marks,
            labels["tolerance_seconds"]) for bvid, marks in labels["videos"].items()]
        checks = [check for video in videos for check in video["boundary_checks"]]
        feasible = [check for check in checks if check["length_feasible"]]
        rate = sum(check["hit"] for check in feasible) / len(feasible) if feasible else 0.0
        passed = all(v["length_passed"] for v in videos) and rate >= 0.8
        report = {"model": settings.embedding_model, "space_id": models.embedding.space_id,
                  "policy": CHUNK_POLICY, "label_status": labels["label_status"],
                  "tolerance_seconds": labels["tolerance_seconds"],
                  "labels": len(checks), "length_constrained_labels": len(checks) - len(feasible),
                  "boundary_hit_rate": rate, "target_passed": passed,
                  "seconds": round(time.monotonic() - started, 3),
                  "model_calls": models.calls.events, "videos": videos}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "target_passed": passed,
                          "boundary_hit_rate": rate, "labels": len(checks),
                          "length_constrained_labels": len(checks) - len(feasible),
                          "chunk_lengths": [v["chunk_lengths"] for v in videos]}, ensure_ascii=False))
        return 0 if passed else 1
    finally:
        for close in models.closers:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
