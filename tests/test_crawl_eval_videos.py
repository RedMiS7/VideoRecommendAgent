import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_agent.config import Settings
from video_agent.demo import demo_videos
from video_agent.models.contracts import CallLog, ModelError
from video_agent.schemas import Profile, Segment
from video_agent.sources import SourceError


spec = importlib.util.spec_from_file_location(
    "crawl_eval_videos", Path(__file__).resolve().parents[1] / "scripts" / "crawl_eval_videos.py")
crawler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crawler)


@pytest.fixture
def services(monkeypatch):
    state = SimpleNamespace(videos=demo_videos(80), queries=[], fetched=[], closed=[],
                            metadata_error=None, subtitle_error=None, model_error=None,
                            assessments={}, screens={}, query_inputs=[], query_prompts=[],
                            screen_inputs=[],
                            foreign_evidence=False)

    class Model:
        def generate_json(self, messages, schema):
            context = json.loads(messages[-1].content)
            task = messages[0].content.splitlines()[0]
            if task == "task:queries":
                if state.model_error:
                    raise state.model_error
                assert context["profile"]["goal"] == "准备 Agent 开发实习"
                state.query_inputs.append(context)
                state.query_prompts.append(messages[0].content)
                offset = len(context["previous_queries"])
                return {"queries": [f"Agent 开发教程 {offset + i}" for i in range(3)]}
            if task == "task:screen":
                state.screen_inputs.append(context)
                return {"items": [{"bvid": v["bvid"], "score": 0.8, "matched": True,
                                   "reason": "主题相关", "evidence_ids": [],
                                   **state.screens.get(v["bvid"], {})}
                                  for v in context["candidates"]]}
            assert task == "task:corpus_quality"
            bvid = context["video"]["bvid"]
            return {"on_topic": True, "substantive": True, "suitable": True,
                    "quality_score": 0.8, "fit_score": 0.85, "mismatch": "none",
                    "reason": "字幕包含具体实现步骤，适合当前学习目标",
                    "evidence_ids": ["foreign" if state.foreign_evidence else
                                     context["chunks"][0]["id"]],
                    **state.assessments.get(bvid, {})}

    class Search:
        def __init__(self, settings):
            pass

        def search(self, query):
            state.queries.append(query)
            return state.videos + state.videos[:2]

        def close(self):
            state.closed.append("search")

    class Source:
        def __init__(self, settings):
            pass

        def metadata(self, video):
            state.fetched.append(video.bvid)
            if state.metadata_error and len(state.fetched) == 2:
                raise state.metadata_error
            return video

        def subtitles(self, video):
            if state.subtitle_error:
                return video.model_copy(update={"subtitle_status": "error", "segments": [],
                                                "subtitle_error": state.subtitle_error})
            return video

        def close(self):
            state.closed.append("source")

    models = SimpleNamespace(language=Model(), calls=CallLog(),
                             closers=[lambda: state.closed.append("model")])
    monkeypatch.setattr(crawler, "build_models", lambda _: models)
    monkeypatch.setattr(crawler, "EnhancedSearch", Search)
    monkeypatch.setattr(crawler, "BilibiliSource", Source)
    monkeypatch.setattr(crawler.time, "sleep", lambda _: None)
    return state


def crawl(tmp_path):
    return crawler.crawl_videos(Settings(), Profile(topics=["Agent 开发"],
                               goal="准备 Agent 开发实习"), tmp_path / "corpus")


def test_crawler_collects_30_unique_subtitled_videos_and_freezes_provenance(tmp_path, services):
    result = crawl(tmp_path)
    assert result["complete"] is True and result["video_count"] == 30
    path = Path(result["videos"])
    videos = json.loads(path.read_text(encoding="utf-8"))
    assert len({video["bvid"] for video in videos}) == 30
    assert all(video["subtitle_status"] == "available" and video["segments"] for video in videos)
    assert len(services.fetched) > 30  # No-subtitle candidates are skipped and replaced.
    assert len(services.fetched) == len(set(services.fetched))
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["source"] == "bilibili" and manifest["requested"] == 30
    assert manifest["videos_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["subtitle_available_count"] == 30
    assert manifest["qualified_pool_count"] >= 45
    assert manifest["positive_candidate_count"] == 30
    assert manifest["label_status"] == "needs_human_review"
    assert (path.parent / "profile.json").exists()
    assert len((path.parent / "bvids.txt").read_text(encoding="utf-8").splitlines()) == 30
    assert set(services.closed) == {"search", "source", "model"}


def test_http_412_preserves_partial_data_and_stops_requests(tmp_path, services):
    services.metadata_error = SourceError("unavailable", "Bilibili HTTP 412")
    result = crawl(tmp_path)
    assert result["video_count"] == 1 and result["complete"] is False
    assert len(services.fetched) == 2 and len(services.queries) == 1
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["stop_reason"] == "Bilibili HTTP 412"
    assert manifest["checks"][-1]["status"] == "error"
    assert set(services.closed) == {"search", "source", "model"}


def test_login_error_stops_and_does_not_count_failed_subtitles(tmp_path, services):
    services.subtitle_error = "Subtitle access requires a valid Bilibili login"
    result = crawl(tmp_path)
    assert result["video_count"] == 0 and result["complete"] is False
    assert len(services.fetched) == 1


def test_candidate_exhaustion_is_bounded_and_does_not_fill_with_duplicates(tmp_path, services):
    services.videos = demo_videos(2)
    result = crawl(tmp_path)
    assert result["video_count"] == 2 and result["complete"] is False
    assert len(services.queries) == crawler.QUERY_BATCHES * 3 and len(services.fetched) == 2
    assert services.query_inputs[1]["search_feedback"][0]["candidate_count"] == 2


def test_query_failure_still_writes_collection_failure_manifest(tmp_path, services):
    services.model_error = ModelError("unavailable", "Query generation failed")
    result = crawl(tmp_path)
    assert result["video_count"] == 0 and result["complete"] is False
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["discovery_errors"][0]["stage"] == "queries"
    assert set(services.closed) == {"search", "source", "model"}


def test_existing_snapshot_is_not_overwritten(tmp_path, services):
    output = tmp_path / "corpus"
    output.mkdir()
    original = b"existing snapshot"
    (output / "videos.json").write_bytes(original)
    with pytest.raises(ValueError, match="already exists"):
        crawl(tmp_path)
    assert (output / "videos.json").read_bytes() == original
    assert services.queries == services.fetched == services.closed == []


def test_actual_content_filters_promotional_and_unrelated_candidates(tmp_path, services):
    services.videos[0].part_title = "实战全套课程"
    services.assessments[services.videos[0].bvid] = {
        "substantive": False, "quality_score": 0.2, "reason": "字幕仅宣传后续课程"}
    services.assessments[services.videos[1].bvid] = {
        "on_topic": False, "reason": "字幕实际讲解无关主题"}
    services.screens[services.videos[2].bvid] = {"matched": False, "score": 0.1}
    services.videos[3].part_title = "课程介绍"
    services.assessments[services.videos[3].bvid] = {"fit_score": 0.99}
    result = crawl(tmp_path)
    collected = json.loads(Path(result["videos"]).read_text(encoding="utf-8"))
    ids = {v["bvid"] for v in collected}
    assert not ids.intersection(v.bvid for v in services.videos[:3])
    assert services.videos[2].bvid not in services.fetched
    assert services.videos[3].bvid in ids  # An intro title alone is not a rejection rule.
    assert result["complete"] is True
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["checks"][0]["reason"] == "字幕仅宣传后续课程"
    assert manifest["checks"][0]["evidence"]
    assert all(word not in services.query_prompts[0]
               for word in ("LangGraph", "LangChain", "MCP", "Text2SQL"))


def test_initial_screen_receives_complete_profile(tmp_path, services):
    profile = Profile(topics=["Agent 开发"], goal="准备 Agent 开发实习",
                      level="已完成推荐 MVP", max_duration_minutes=30,
                      preferences=[{"dimension": "style", "value": "practical",
                                    "direction": "prefer", "scope": "global"}])
    crawler.crawl_videos(Settings(), profile, tmp_path / "corpus")
    assert services.screen_inputs
    assert all(item["context"]["profile"] == profile.model_dump()
               for item in services.screen_inputs)


def test_hard_negatives_have_substantive_content_and_are_capped_at_two(tmp_path, services):
    available = [v for v in services.videos if v.segments]
    for video in available[28:]:
        services.assessments[video.bvid] = {
            "suitable": False, "fit_score": 0.4, "mismatch": "level",
            "reason": "主题相关、讲解充实，但内容过于基础"}
    result = crawl(tmp_path)
    assert result["video_count"] == 30 and result["complete"] is True
    assert result["positive_candidate_count"] == 28
    assert result["hard_negative_candidate_count"] == 2
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    negatives = [c for c in manifest["checks"] if c["accepted"] and
                 c["pool_role"] == "hard_negative_candidate"]
    assert len(negatives) == 2
    assert all(c["assessment"]["substantive"] and c["evidence"] for c in negatives)
    assert manifest["selection_policy"]["max_hard_negatives"] == 2
    assert manifest["selection_policy"]["max_hard_negative_ratio"] == 0.1


@pytest.mark.parametrize("positive_count", [0, 2, 8, 9, 18])
def test_negative_share_is_bounded_in_partial_corpora(tmp_path, services, positive_count):
    services.videos = demo_videos(28)
    available = [v for v in services.videos if v.segments]
    for video in available[positive_count:]:
        services.assessments[video.bvid] = {
            "suitable": False, "fit_score": 0.4, "mismatch": "style"}
    result = crawl(tmp_path)
    negative_count = min(2, positive_count // 9)
    assert result["video_count"] == positive_count + negative_count
    assert result["positive_candidate_count"] == positive_count
    assert result["hard_negative_candidate_count"] == negative_count
    assert result["complete"] is False
    assert not result["video_count"] or negative_count / result["video_count"] <= 0.1


def test_content_evidence_must_belong_to_the_assessed_video(tmp_path, services):
    services.videos = demo_videos(1)
    services.foreign_evidence = True
    result = crawl(tmp_path)
    assert result["video_count"] == 0 and result["complete"] is False
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["checks"][0]["status"] == "assessment_error"


def test_candidate_budget_bounds_source_requests(tmp_path, services):
    services.videos = demo_videos(150)
    for video in services.videos:
        services.assessments[video.bvid] = {"substantive": False, "quality_score": 0.2}
    result = crawl(tmp_path)
    assert result["attempted"] == crawler.MAX_CANDIDATES
    assert len(services.fetched) == crawler.MAX_CANDIDATES
    assert result["complete"] is False


def test_final_selection_can_replace_an_early_weaker_candidate(tmp_path, services):
    services.assessments[services.videos[0].bvid] = {"fit_score": 0.7}
    services.assessments[services.videos[50].bvid] = {"fit_score": 0.99}
    result = crawl(tmp_path)
    collected = json.loads(Path(result["videos"]).read_text(encoding="utf-8"))
    ids = {v["bvid"] for v in collected}
    assert services.videos[0].bvid not in ids
    assert services.videos[50].bvid in ids


def test_low_quality_or_unexplained_mismatch_is_not_a_hard_negative(tmp_path, services):
    services.videos = demo_videos(2)
    services.assessments[services.videos[0].bvid] = {
        "suitable": False, "fit_score": 0.4, "quality_score": 0.2, "mismatch": "style"}
    services.assessments[services.videos[1].bvid] = {"suitable": False, "fit_score": 0.4}
    result = crawl(tmp_path)
    assert result["video_count"] == 0 and result["hard_negative_candidate_count"] == 0


def test_long_subtitle_review_covers_beginning_middle_and_end_with_a_bound():
    video = demo_videos(1)[0]
    video.segments = [Segment(start=i, end=i + 1, text=f"字幕{i}:" + "内容" * 600)
                      for i in range(40)]
    chunks, coverage = crawler._review_chunks(video)
    assert coverage["coverage"] == "sampled"
    assert coverage["assessed_chars"] <= crawler.CHUNK_CHARS * crawler.MAX_REVIEW_CHUNKS
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == 40
    assert any(18 <= c["start"] <= 22 for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)


@pytest.mark.parametrize("wrapped", [False, True])
def test_profile_loader_accepts_current_profile_file_and_single_case(tmp_path, wrapped):
    profile = {"topics": ["Agent 开发"], "goal": "准备实习"}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"profile": profile} if wrapped else profile), encoding="utf-8")
    assert crawler.load_profile(path).goal == "准备实习"
