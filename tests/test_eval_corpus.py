import json

import pytest

from video_agent.config import Settings
from video_agent.demo import demo_videos
from video_agent.eval_corpus import collect_corpus
from video_agent.sources import SourceError


def test_collection_freezes_available_and_absent_subtitles_without_models(tmp_path, monkeypatch):
    videos = {v.bvid: v for v in demo_videos(7)}
    closed = []

    class Source:
        def __init__(self, settings):
            pass

        def metadata(self, seed):
            return videos[seed.bvid]

        def subtitles(self, video):
            return video

        def close(self):
            closed.append(True)

    monkeypatch.setattr("video_agent.eval_corpus.BilibiliSource", Source)
    seeds = tmp_path / "bvids.txt"
    seeds.write_text("BV0000000001\nBV0000000007\nBV0000000001\n", encoding="utf-8")
    output = tmp_path / "corpus"
    result = collect_corpus(Settings(), seeds, output)
    assert result["video_count"] == 2 and result["subtitle_available_count"] == 1
    assert result["requested"] == 2 and closed == [True]
    assert result["complete"] is True
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "bilibili" and manifest["videos_sha256"]
    original = (output / "videos.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        collect_corpus(Settings(), seeds, output)
    assert (output / "videos.json").read_bytes() == original


def test_collection_stops_on_http_412_and_keeps_partial_snapshot(tmp_path, monkeypatch):
    calls = []
    closed = []

    class Source:
        def __init__(self, settings):
            pass

        def metadata(self, seed):
            calls.append(seed.bvid)
            if len(calls) == 2:
                raise SourceError("unavailable", "Bilibili HTTP 412")
            return demo_videos(1)[0]

        def subtitles(self, video):
            return video

        def close(self):
            closed.append(True)

    monkeypatch.setattr("video_agent.eval_corpus.BilibiliSource", Source)
    monkeypatch.setattr("video_agent.eval_corpus.time.sleep", lambda _: None)
    seeds = tmp_path / "bvids.txt"
    seeds.write_text("\n".join(f"BV{i:010d}" for i in range(1, 4)), encoding="utf-8")
    output = tmp_path / "corpus"
    result = collect_corpus(Settings(), seeds, output)
    assert result["attempted"] == 2 and result["video_count"] == 1
    assert result["complete"] is False
    assert calls == ["BV0000000001", "BV0000000002"] and closed == [True]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checks"][1]["status"] == "error"


def test_collection_stops_on_invalid_login(tmp_path, monkeypatch):
    calls = []

    class Source:
        def __init__(self, settings):
            pass

        def metadata(self, seed):
            calls.append(seed.bvid)
            return demo_videos(1)[0]

        def subtitles(self, video):
            return video.model_copy(update={"subtitle_status": "error", "segments": [],
                                            "subtitle_error": "refresh login"})

        def close(self):
            pass

    monkeypatch.setattr("video_agent.eval_corpus.BilibiliSource", Source)
    seeds = tmp_path / "bvids.txt"
    seeds.write_text("BV0000000001\nBV0000000002\n", encoding="utf-8")
    result = collect_corpus(Settings(), seeds, tmp_path / "corpus")
    assert result["attempted"] == 1 and result["subtitle_available_count"] == 0
    assert len(calls) == 1
