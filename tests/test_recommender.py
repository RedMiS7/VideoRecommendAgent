from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

from video_agent.demo import DemoSource, demo_videos
from video_agent.models.contracts import ModelError
from video_agent.recommender import score_candidates
from video_agent.runtime import Runtime
from video_agent.config import Settings
from video_agent.schemas import FeedbackInput, Profile, utc_now
from video_agent.sources import SourceError
from video_agent.store import RunBusy, Store


@pytest.fixture
def runtime(tmp_path):
    result = Runtime(Settings(data_dir=tmp_path), offline=True)
    result.store.save_profile(Profile(topics=["RAG"]))
    yield result
    result.close()


def test_eight_days_have_no_recent_duplicates_and_same_day_is_idempotent(runtime):
    now = utc_now()
    days = []
    for day in range(8):
        at = now + timedelta(days=day)
        result = runtime.recommender.run(at)
        ids = {item["bvid"] for item in result["items"]}
        assert len(ids) == 5
        for previous_day, previous in days:
            if (day - previous_day) < 7:
                assert not ids & previous
        assert runtime.recommender.run(at) == result
        days.append((day, ids))


def test_no_subtitle_only_fills_remaining_slots_and_shortage_is_reported(runtime):
    videos = demo_videos(2)
    videos[0].subtitle_status, videos[0].segments = "absent", []
    source = DemoSource(videos)
    runtime.recommender.search = runtime.recommender.source = source
    result = runtime.recommender.run()
    assert len(result["items"]) == 2
    assert result["items"][0]["subtitle_status"] == "available"
    assert result["items"][1]["subtitle_status"] == "absent"
    assert "标题和摘要" in result["items"][1]["basis"]
    assert result["items"][1]["evidence"] == []
    assert any("Only 2" in warning for warning in result["warnings"])


def test_search_failure_uses_cached_candidates(runtime):
    for video in demo_videos(2):
        runtime.store.save_video(video)

    class FailedSearch:
        def search(self, query):
            raise SourceError("unavailable", "Search unavailable")

    runtime.recommender.search = FailedSearch()
    result = runtime.recommender.run()
    assert len(result["items"]) == 2
    assert any("cached" in warning for warning in result["warnings"])


def test_empty_success_is_retried_when_search_recovers(runtime):
    runtime.recommender.search = runtime.recommender.source = DemoSource([])
    now = utc_now()
    first = runtime.recommender.run(now)
    assert first["items"] == []
    assert runtime.store.run(first["date"])["status"] == "success"

    runtime.recommender.search = runtime.recommender.source = DemoSource(demo_videos(2))
    second = runtime.recommender.run(now + timedelta(seconds=1))
    assert len(second["items"]) == 2
    assert runtime.recommender.run(now + timedelta(seconds=2)) == second


def test_model_failure_is_persisted_and_not_fabricated(runtime):
    class FailedModel:
        def generate_json(self, messages, schema):
            raise ModelError("unavailable", "Model unavailable")

    runtime.recommender.model = FailedModel()
    with pytest.raises(ModelError):
        runtime.recommender.run()
    date = utc_now().astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()
    record = runtime.store.run(date)
    assert record["status"] == "failed" and record["payload"] is None


def test_unknown_duration_exclusion_is_explained(runtime):
    videos = demo_videos(2)
    for video in videos:
        video.duration = None
    runtime.store.save_profile(Profile(topics=["RAG"], max_duration_minutes=30))
    runtime.recommender.search = runtime.recommender.source = DemoSource(videos)
    result = runtime.recommender.run()
    assert result["items"] == []
    assert any("2 candidates excluded: duration unavailable" in w for w in result["warnings"])


def test_ban_and_watched_exclude_videos(runtime):
    video = demo_videos(1)[0]
    runtime.store.save_video(video)
    runtime.memory.submit(FeedbackInput(bvid=video.bvid, text="看完了，不再推荐这个"))
    assert video.bvid in runtime.store.excluded(utc_now())


def test_cross_video_evidence_is_rejected():
    class BadModel:
        def generate_json(self, messages, schema):
            return {"items": [{"bvid": "BV0000000001", "matched": True, "score": 1,
                               "reason": "wrong", "evidence_ids": ["other-video"]}]}

    with pytest.raises(ModelError):
        score_candidates(BadModel(), [{"bvid": "BV0000000001", "chunks": [{"id": "own"}]}], {})


def test_atomic_claim_across_connections(tmp_path):
    path = tmp_path / "agent.sqlite"
    a, b = Store(path), Store(path)
    now = utc_now()
    a.claim_run("2026-10-01", now)
    with pytest.raises(RunBusy):
        b.claim_run("2026-10-01", now)
    a.close()
    b.close()


def test_expired_worker_cannot_publish_over_new_owner(tmp_path):
    a, b = Store(tmp_path / "agent.sqlite"), Store(tmp_path / "agent.sqlite")
    date, now = "2026-10-01", utc_now()
    later = now + timedelta(hours=1, seconds=1)
    a.claim_run(date, now)
    b.claim_run(date, later)
    with pytest.raises(RunBusy):
        a.complete_run(date, {"items": []}, now)
    a.fail_run(date, "old worker failed", now)
    assert b.run(date)["status"] == "running"
    b.complete_run(date, {"items": []}, later)
    assert b.run(date)["status"] == "success"
    a.close()
    b.close()


def test_single_global_feedback_does_not_affect_other_videos(runtime):
    from video_agent.recommender import brief

    videos = demo_videos(5)
    for video in videos:
        runtime.store.save_video(video)
    runtime.memory.submit(FeedbackInput(bvid=videos[0].bvid, text="以后不要太基础的内容"))
    assert not runtime.store.profile().preferences
    result = score_candidates(runtime.models.language, [brief(v) for v in videos],
                              runtime.memory.context(), "screen")
    by_id = {s.bvid: s for s in result.items}
    assert not by_id[videos[0].bvid].matched
    assert by_id[videos[2].bvid].matched
    assert by_id[videos[4].bvid].matched
    for video in [videos[2], videos[4]]:
        runtime.memory.submit(FeedbackInput(bvid=video.bvid, text="以后不要太基础的内容"))
    assert runtime.store.profile().preferences
    next_day = runtime.recommender.run(utc_now() + timedelta(days=1))
    assert all("基础" not in item["title"] for item in next_day["items"])
