import json
import math
from pathlib import Path

import pytest

from video_agent.config import Settings
from video_agent.evaluation import load_dataset, run_evaluation
from video_agent.models.contracts import ModelError
from video_agent.recommendation_evaluation import recommendation_metrics, usage_cost
from video_agent.recommender import Recommender


def test_precision_and_ndcg_use_human_grades_and_full_candidate_pool():
    relevance = {"a": 3, "b": 2, "c": 2, "d": 1, "e": 0, "f": 3}
    items = [{"bvid": bvid} for bvid in ["d", "a", "e", "b", "c"]]
    scores = recommendation_metrics(items, relevance, {})
    dcg = 1 + 7 / math.log2(3) + 3 / math.log2(5) + 3 / math.log2(6)
    ideal = 7 + 7 / math.log2(3) + 3 / math.log2(4) + 3 / math.log2(5) + 1 / math.log2(6)
    assert scores["precision_at_5"] == 0.6
    assert scores["ndcg_at_5"] == pytest.approx(dcg / ideal)
    ideal_items = [{"bvid": bvid} for bvid in ["a", "f", "b", "c", "d"]]
    assert recommendation_metrics(ideal_items, relevance, {})["ndcg_at_5"] == 1


def test_short_list_does_not_inflate_precision_and_no_ideal_is_undefined():
    scores = recommendation_metrics([{"bvid": "a"}], {"a": 3, "b": 2}, {"a": [1, 0]})
    assert scores["precision_at_5"] == 0.2
    assert scores["ndcg_at_5"] < 1
    assert scores["diversity_at_5"] is None
    empty = recommendation_metrics([], {"a": 0}, {})
    assert empty == {"precision_at_5": 0, "ndcg_at_5": None, "diversity_at_5": None}
    pending = recommendation_metrics([], {"a": None}, {})
    assert pending["precision_at_5"] is pending["ndcg_at_5"] is None


def test_diversity_is_mean_pairwise_cosine_distance_and_requires_all_vectors():
    items = [{"bvid": bvid} for bvid in ["a", "b", "c"]]
    vectors = {"a": [2, 0], "b": [1, 0], "c": [0, 1]}
    scores = recommendation_metrics(items, {bvid: 2 for bvid in vectors}, vectors)
    assert scores["diversity_at_5"] == pytest.approx(2 / 3)
    assert recommendation_metrics(items[:2], {"a": 2, "b": 2}, vectors)["diversity_at_5"] == 0
    assert recommendation_metrics(items, {bvid: 2 for bvid in vectors},
                                  {"a": [1, 0], "b": [1, 0]})["diversity_at_5"] is None
    assert recommendation_metrics(items[:2], {"a": 2, "b": 2},
                                  {"a": [0, 0], "b": [1, 0]})["diversity_at_5"] is None


def test_cost_includes_retry_calls_and_embedding_and_does_not_guess_missing_usage():
    pricing = {"currency": "CNY", "models": {
        "text": {"input_per_million": 2, "output_per_million": 4},
        "embedding": {"input_per_million": 1, "output_per_million": 0}}}
    calls = [
        {"model": "text", "operation": "/chat/completions", "status": "success",
         "usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
        {"model": "text", "operation": "/chat/completions", "status": "success",
         "usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
        {"model": "embedding", "operation": "/embeddings", "status": "success",
         "usage": {"total_tokens": 1000}},
    ]
    result = usage_cost(calls, pricing)
    assert result["input_tokens"] == 3000 and result["output_tokens"] == 1000
    assert result["estimated_model_cost"] == pytest.approx(0.009)
    calls.append({"model": "text", "status": "failed", "usage": None})
    assert usage_cost(calls, pricing)["estimated_model_cost"] is None
    assert usage_cost(calls[:-1], None)["estimated_model_cost"] is None


def write_dataset(tmp_path, cases):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    (tmp_path / "videos.json").write_bytes(Path("evals/videos.json").read_bytes())
    (tmp_path / "manifest.json").write_text('{"source":"synthetic"}', encoding="utf-8")
    return dataset


def test_default_suite_runs_actual_recommender_with_isolated_profiles(tmp_path, monkeypatch):
    run = Recommender.run
    profiles, paths = [], []

    def capture(self, *args, **kwargs):
        profiles.append(self.store.profile().model_dump())
        paths.append(self.store.connection.execute("PRAGMA database_list").fetchone()[2])
        return run(self, *args, **kwargs)

    monkeypatch.setattr(Recommender, "run", capture)
    result = run_evaluation(Settings(data_dir=tmp_path / "data"),
                            Path("evals/recommendation-cases.jsonl"),
                            tmp_path / "reports", offline=True)
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert len(profiles) == 4 and all(p["daily_count"] == 5 for p in profiles)
    assert len(set(paths)) == 4
    assert report["mode"] == "offline_test_doubles"
    assert report["label_review"]["pending"] == 4
    assert set(report["metrics"]) == {"precision_at_5", "ndcg_at_5", "diversity_at_5", "cost_latency"}
    assert len(report["cases"][0]["items"]) == 5
    assert all("judge" not in record for record in report["cases"])
    assert report["metrics"]["cost_latency"]["mean_estimated_model_cost"] is None
    assert not (tmp_path / "data" / "agent.sqlite").exists()
    assert Path(result["markdown_report"]).exists()


def test_failures_stay_in_precision_and_ndcg_denominators(tmp_path, monkeypatch):
    case = json.loads(Path("evals/recommendation-cases.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    calls = 0
    run = Recommender.run

    def fail_once(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelError("unavailable", "test outage")
        return run(self)

    monkeypatch.setattr(Recommender, "run", fail_once)
    result = run_evaluation(Settings(data_dir=tmp_path / "data"),
                            write_dataset(tmp_path, [case, {**case, "id": "second"}]),
                            tmp_path / "reports", offline=True)
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert report["cases"][0]["status"] == "error"
    assert report["cases"][1]["status"] == "success"
    assert report["metric_samples"]["precision_at_5"] == {"valid": 2, "total": 2}
    assert report["metrics"]["precision_at_5"] == report["cases"][1]["metrics"]["precision_at_5"] / 2
    assert report["metric_samples"]["ndcg_at_5"]["valid"] == 2


@pytest.mark.parametrize("change", ["missing", "out_of_range", "bool", "reviewed_null", "count"])
def test_dataset_requires_complete_valid_relevance_labels(tmp_path, change):
    case = json.loads(Path("evals/recommendation-cases.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    key = next(iter(case["relevance"]))
    if change == "missing":
        case["relevance"].pop(key)
    elif change == "out_of_range":
        case["relevance"][key] = 4
    elif change == "bool":
        case["relevance"][key] = True
    elif change == "reviewed_null":
        case["label_status"], case["relevance"][key] = "human_reviewed", None
    else:
        case["profile"]["daily_count"] = 3
    with pytest.raises(ValueError):
        load_dataset(write_dataset(tmp_path, [case]))
