import hashlib
import json
import time
from pathlib import Path

from .schemas import Video, utc_now
from .sources import BilibiliSource, SourceError


def collect_corpus(settings, bvid_file: Path, output: Path) -> dict:
    bvids = list(dict.fromkeys(line.strip() for line in
                              bvid_file.read_text(encoding="utf-8-sig").splitlines()
                              if line.strip()))
    if not bvids:
        raise ValueError("Provide at least one BVID")
    seeds = [Video(bvid=bvid, title="Collection pending") for bvid in bvids]
    if any((output / name).exists() for name in ("videos.json", "manifest.json")):
        raise ValueError("Corpus snapshot already exists; choose a new output directory")
    source = BilibiliSource(settings)
    videos, checks = [], []
    try:
        for index, seed in enumerate(seeds):
            if index:
                time.sleep(1)
            try:
                video = source.subtitles(source.metadata(seed))
                videos.append(video.model_dump(mode="json"))
                checks.append({"bvid": video.bvid, "status": video.subtitle_status,
                               "segments": len(video.segments), "error": video.subtitle_error})
                if video.subtitle_status == "error" and (
                    "login" in (video.subtitle_error or "") or
                    "BILI_SESSDATA" in (video.subtitle_error or "") or
                    "HTTP 412" in (video.subtitle_error or "")
                ):
                    break
            except SourceError as error:
                checks.append({"bvid": seed.bvid, "status": "error",
                               "kind": error.kind, "error": str(error)})
                if "HTTP 412" in str(error):
                    break
    finally:
        source.close()
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "videos.json"
    video_path.write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")
    available = sum(v["subtitle_status"] == "available" for v in videos)
    manifest = {"version": "real-v1", "source": "bilibili", "collected_at": utc_now().isoformat(),
                "requested": len(seeds), "attempted": len(checks), "video_count": len(videos),
                "subtitle_available_count": available, "checks": checks,
                "videos_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest()}
    manifest["complete"] = len(checks) == len(seeds) and not any(
        check["status"] == "error" for check in checks)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"videos": str(video_path), "manifest": str(output / "manifest.json"),
            "video_count": len(videos), "subtitle_available_count": available,
            "requested": len(seeds), "attempted": len(checks), "complete": manifest["complete"]}
