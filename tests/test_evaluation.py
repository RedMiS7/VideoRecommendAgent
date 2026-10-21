import json
from pathlib import Path

import pytest

from video_agent.config import Settings
from video_agent.evaluation import load_dataset, run_evaluation
from video_agent.evaluation_reports import compute_metrics, refresh_report, review_evaluation
from video_agent.models.contracts import ModelError
from video_agent.rag import Rag
from video_agent.schemas import Profile


def write_dataset(tmp_path, cases):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    (tmp_path / "videos.json").write_bytes(Path("evals/videos.json").read_bytes())
    return dataset


def test_no_memory_retrieval_does_not_receive_preferences(tmp_path, monkeypatch):
    case = json.loads(Path("evals/cases.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    dataset = write_dataset(tmp_path, [case])
    queries = []
    retrieve = Rag.retrieve

    def capture(self, query, *args, **kwargs):
        queries.append(json.loads(query))
        return retrieve(self, query, *args, **kwargs)

    monkeypatch.setattr(Rag, "retrieve", capture)
    run_evaluation(Settings(data_dir=tmp_path / "data"), dataset,
                   tmp_path / "reports", offline=True)
    assert len(queries) == 2
    assert queries[0]["preferences"] == []
    assert queries[1]["preferences"] == Profile.model_validate(case["profile"]).model_dump()["preferences"]


def test_no_match_accepts_empty_result_and_filters_low_scores(tmp_path, monkeypatch):
    from video_agent.demo import DemoLanguageModel

    generate = DemoLanguageModel.generate_json

    def low_scores(self, messages, schema):
        result = generate(self, messages, schema)
        if messages[0].content.startswith("task:rank"):
            for score in result["items"]:
                score["matched"], score["score"] = True, 0.49
        return result

    monkeypatch.setattr(DemoLanguageModel, "generate_json", low_scores)
    dataset = write_dataset(tmp_path, [{"id": "no-match", "type": "ranking",
                                      "profile": {"topics": ["RAG"]}, "gold_bvids": []}])
    result = run_evaluation(Settings(data_dir=tmp_path / "data"), dataset,
                            tmp_path / "reports", offline=True)
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert all(v["top_bvid"] is None and v["passed"]
               for v in report["cases"][0]["variants"].values())


def test_offline_evaluation_isolated_and_does_not_claim_ai_quality(tmp_path):
    report = run_evaluation(Settings(data_dir=tmp_path / "data"), Path("evals/cases.jsonl"),
                            tmp_path / "reports", offline=True)
    assert report["case_count"] == 30 and report["quality_verified"] is False
    assert not (tmp_path / "data" / "agent.sqlite").exists()
    details = json.loads(Path(report["json_report"]).read_text(encoding="utf-8"))
    assert len(details["cases"]) == 30
    assert details["metrics"]["human_support_rate"] is None
    assert details["metrics"]["chunk_hit_at_5"] is not None
    assert Path(report["markdown_report"]).exists()
    assert details["prompt_hashes"] and details["dataset_manifest"]["cases_sha256"]
    assert details["metrics"]["video_hit_at_5"] == 1
    assert details["human_review"]["pending"] > 0


@pytest.fixture
def evaluation_report(tmp_path):
    cases = [json.loads(line) for line in Path("evals/cases.jsonl").read_text(
        encoding="utf-8").splitlines()]
    selected = [cases[10], cases[11], cases[20], cases[21]]
    for case in selected:
        case.update(split="test", label_status="human_reviewed")
    result = run_evaluation(Settings(data_dir=tmp_path / "data"),
                            write_dataset(tmp_path, selected), tmp_path / "reports", offline=True)
    return Path(result["json_report"])


def test_metrics_count_failed_retrieval_and_rank_position():
    records = [
        {"id": "hit", "type": "retrieval", "status": "success", "gold": "b",
         "retrieved": ["a", "b"], "passed": True, "chunk_passed": True},
        {"id": "failed", "type": "retrieval", "status": "error", "gold": "a",
         "retrieved": [], "passed": False, "chunk_passed": False},
    ]
    metrics, samples = compute_metrics(records)
    assert metrics["video_hit_at_1"] == 0
    assert metrics["video_hit_at_5"] == 0.5
    assert metrics["video_mrr_at_5"] == 0.25
    assert metrics["execution_failure_rate"] == 0.5
    assert samples["chunk_hit_at_5"] == {"numerator": 1, "denominator": 2}


def test_service_failure_is_retained_and_other_cases_continue(tmp_path, monkeypatch):
    retrieve = Rag.retrieve

    def fail(self, query, *args, **kwargs):
        if query == "service fails":
            raise ModelError("unavailable", "test outage")
        return retrieve(self, query, *args, **kwargs)

    monkeypatch.setattr(Rag, "retrieve", fail)
    case = {"id": "failed", "type": "retrieval", "query": "service fails",
            "gold_bvid": "BV0000000001", "gold_start": 0, "gold_end": 60}
    dataset = write_dataset(tmp_path, [case, {**case, "id": "ok", "query": "RAG"}])
    result = run_evaluation(Settings(data_dir=tmp_path / "data"), dataset,
                            tmp_path / "reports", offline=True)
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert report["cases"][0]["status"] == "error"
    assert report["cases"][1]["status"] == "success"
    assert report["metric_samples"]["video_hit_at_5"]["denominator"] == 2
    assert report["metrics"]["execution_failure_rate"] == 0.5


def test_index_failure_preserves_report_and_blocks_rag(tmp_path, monkeypatch):
    monkeypatch.setattr(Rag, "index_video", lambda *args: (_ for _ in ()).throw(
        ModelError("unavailable", "embedding outage")))
    case = json.loads(Path("evals/cases.jsonl").read_text(encoding="utf-8").splitlines()[20])
    result = run_evaluation(Settings(data_dir=tmp_path / "data"),
                            write_dataset(tmp_path, [case]), tmp_path / "reports", offline=True)
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert len(report["index_errors"]) == 6
    assert report["cases"][0]["variants"]["rag_memory"]["status"] == "error"
    assert report["quality_verified"] is False


def read_review(report):
    result = review_evaluation(report)
    path = Path(result["review_template"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return path, rows


def test_review_roundtrip_preserves_original_and_calibrates_judge(evaluation_report):
    original = evaluation_report.read_bytes()
    path, rows = read_review(evaluation_report)
    assert len(rows) == 4
    assert all(row["supported"] is None and "judge" not in row for row in rows)
    assert all(not row["profile"]["preferences"] for row in rows
               if row["variant"] == "rag_no_memory")
    for row in rows:
        row.update(supported=row["variant"] == "rag_memory", note="Human test label")
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = review_evaluation(evaluation_report, path)
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert evaluation_report.read_bytes() == original
    assert report["human_review"]["pending"] == 0
    assert report["metrics"]["human_support_rate"] == 1
    assert report["metrics"]["rag_no_memory_human_support_rate"] == 0
    assert report["metrics"]["rag_no_memory_judge_human_agreement"] == 0
    assert report["metrics"]["rag_no_memory_judge_false_support_rate"] == 1
    assert report["quality_verified"] is False  # Offline mode never verifies AI quality.


@pytest.mark.parametrize("change", ["wrong_run", "wrong_hash", "null", "duplicate", "unknown"])
def test_invalid_reviews_are_rejected(evaluation_report, change):
    path, rows = read_review(evaluation_report)
    for row in rows:
        row.update(supported=True, note="review")
    if change == "wrong_run":
        rows[0]["run_id"] = "another run"
    elif change == "wrong_hash":
        rows[0]["output_hash"] = "stale output"
    elif change == "null":
        rows[0]["supported"] = None
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["case_id"] = "unknown"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        review_evaluation(evaluation_report, path)
    assert not evaluation_report.with_name(evaluation_report.stem + ".reviewed.json").exists()


def test_partial_review_does_not_accept_quality(evaluation_report):
    path, rows = read_review(evaluation_report)
    rows[0].update(supported=True, note="one reviewed")
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    result = review_evaluation(evaluation_report, path)
    assert result["human_review"] == {"required": 4, "reviewed": 1, "pending": 3}
    assert result["quality_verified"] is False


def test_quality_gate_uses_test_split_and_requires_all_labels(evaluation_report):
    report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    report["mode"] = "real_models_real_dataset"  # Gate unit test, not real AI verification.
    for case in report["cases"]:
        if case["type"] == "retrieval":
            case["passed"] = case["chunk_passed"] = True
        else:
            for variant, result in case["variants"].items():
                if variant != "title":
                    result["human_supported"] = True
    refresh_report(report)
    assert report["quality_verified"] is True
    report["cases"][0]["chunk_passed"] = False
    refresh_report(report)
    assert report["quality_verified"] is False
    report["cases"][0]["chunk_passed"] = True
    report["cases"][0]["label_status"] = "needs_human_review"
    refresh_report(report)
    assert report["quality_verified"] is False


def test_dataset_rejects_cross_split_intent_and_changed_corpus(tmp_path):
    case = {"id": "one", "type": "ranking", "profile": {"topics": ["RAG"]},
            "gold_bvids": [], "split": "dev", "intent_id": "same question"}
    dataset = write_dataset(tmp_path, [case, {**case, "id": "two", "split": "test"}])
    with pytest.raises(ValueError, match="paraphrases"):
        load_dataset(dataset)
    write_dataset(tmp_path, [case])
    (tmp_path / "manifest.json").write_text(
        '{"source":"bilibili","videos_sha256":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        load_dataset(dataset)


def test_review_export_does_not_overwrite_human_work(evaluation_report):
    path, rows = read_review(evaluation_report)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        review_evaluation(evaluation_report)
    assert path.read_bytes() == original and rows


def test_review_cli_never_constructs_model_runtime(evaluation_report, monkeypatch, capsys):
    from video_agent import cli

    def unexpected_runtime(*args, **kwargs):
        raise AssertionError("Human review must not initialize model or source services")

    monkeypatch.setattr(cli, "Runtime", unexpected_runtime)
    monkeypatch.setattr("sys.argv", ["video-agent", "eval-review",
                                    "--report", str(evaluation_report)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["review_count"] == 4
