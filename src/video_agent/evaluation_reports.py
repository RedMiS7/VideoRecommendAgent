import hashlib
import json
from pathlib import Path

from .schemas import utc_now


VARIANTS = ("title", "rag_no_memory", "rag_memory")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def compute_metrics(records: list[dict]) -> tuple[dict, dict]:
    metrics, samples = {}, {}

    def measure(name, values):
        metrics[name] = sum(values) / len(values) if values else None
        samples[name] = {"numerator": sum(values), "denominator": len(values)}

    retrieval = [r for r in records if r["type"] == "retrieval"]
    ranking = [r for r in records if r["type"] == "ranking"]
    measure("feedback_case_accuracy", [r["passed"] for r in records if r["type"] == "feedback"])
    measure("video_hit_at_1", [r.get("retrieved", [])[:1] == [r["gold"]] for r in retrieval])
    measure("video_hit_at_5", [r["passed"] for r in retrieval])
    measure("video_mrr_at_5", [1 / (r["retrieved"].index(r["gold"]) + 1)
                             if r["gold"] in r.get("retrieved", []) else 0 for r in retrieval])
    measure("chunk_hit_at_5", [r.get("chunk_passed", False) for r in retrieval])
    measure("execution_failure_rate", [r["status"] == "error" for r in records])
    for variant in VARIANTS:
        group = [r["variants"][variant] for r in ranking]
        measure(variant + "_top1_accuracy", [v["passed"] for v in group])
        measure(variant + "_empty_result_rate", [v.get("top_bvid") is None for v in group])
        if variant == "title":
            continue
        outputs = [v for v in group if "reason" in v]
        humans = [v["human_supported"] for v in outputs
                  if isinstance(v.get("human_supported"), bool)]
        judges = [v["judge"]["supported"] for v in outputs if "judge" in v]
        pairs = [v for v in outputs if "judge" in v
                 and isinstance(v.get("human_supported"), bool)]
        measure(variant + "_human_support_rate", humans)
        measure(variant + "_human_review_coverage", [
            isinstance(v.get("human_supported"), bool) for v in outputs])
        measure(variant + "_model_judged_support_rate", judges)
        measure(variant + "_judge_human_agreement", [
            v["judge"]["supported"] == v["human_supported"] for v in pairs])
        # Conditional on the judge claiming support: how often did a human disagree?
        measure(variant + "_judge_false_support_rate", [
            not v["human_supported"] for v in pairs if v["judge"]["supported"]])
    for public, metric in {
        "personalized_top1_accuracy": "rag_memory_top1_accuracy",
        "model_judged_support_rate": "rag_memory_model_judged_support_rate",
        "human_support_rate": "rag_memory_human_support_rate",
    }.items():
        metrics[public], samples[public] = metrics[metric], samples[metric]
    return metrics, samples


def review_outputs(report):
    return [(case, variant, result)
            for case in report["cases"] if case["type"] == "ranking"
            for variant, result in case["variants"].items()
            if variant != "title" and "reason" in result]


def refresh_report(report):
    report["metrics"], report["metric_samples"] = compute_metrics(report["cases"])
    report["metrics_by_split"] = {}
    for split in sorted({c["split"] for c in report["cases"]}):
        records = [c for c in report["cases"] if c["split"] == split]
        metrics, samples = compute_metrics(records)
        report["metrics_by_split"][split] = {"case_count": len(records),
                                            "metrics": metrics, "metric_samples": samples}
    outputs = review_outputs(report)
    reviewed = sum(isinstance(v.get("human_supported"), bool) for _, _, v in outputs)
    report["human_review"] = {"required": len(outputs), "reviewed": reviewed,
                              "pending": len(outputs) - reviewed}
    gold_reviewed = all(c["label_status"] == "human_reviewed" for c in report["cases"])
    report["dataset_label_status"] = "human_reviewed" if gold_reviewed else "needs_human_review"
    test = [c for c in report["cases"] if c["split"] == "test"]
    metrics, _ = compute_metrics(test)
    blockers = []
    if report["mode"] != "real_models_real_dataset":
        blockers.append("Real models and an explicitly sourced real corpus are required")
    if not gold_reviewed:
        blockers.append("Gold labels have not all been human reviewed")
    if not {"retrieval", "ranking"} <= {c["type"] for c in test}:
        blockers.append("A frozen test split containing retrieval and ranking cases is required")
    if report["index_errors"] or any(c["status"] == "error" for c in report["cases"]):
        blockers.append("Unresolved execution failures")
    if reviewed != len(outputs):
        blockers.append("Human output reviews are incomplete")
    for metric in ("chunk_hit_at_5", "human_support_rate"):
        value = metrics[metric]
        if value is None or value < report["targets"][metric]:
            blockers.append(f"Test {metric} has not met {report['targets'][metric]}")
    if any(c["type"] == "feedback" for c in test):
        value = metrics["feedback_case_accuracy"]
        if value is None or value < report["targets"]["feedback_case_accuracy"]:
            blockers.append("Test feedback accuracy has not met its target")
    report["quality_verified"] = not blockers
    report["acceptance"] = {"scope": "this frozen dataset's test split",
                             "blockers": blockers, "test_metrics": metrics}


def write_report(report, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = ["# Video Agent Evaluation", "", f"Run: {report['run_id']}",
                f"Mode: {report['mode']}", f"Cases: {report['case_count']}",
                f"Videos: {report['video_count']}", "",
                f"AI quality acceptance: **{'passed for this test split' if report['quality_verified'] else 'pending / not passed'}**",
                "", "| Metric | Result | Numerator | Denominator |",
                "| --- | --- | --- | --- |"]
    for key, value in report["metrics"].items():
        sample = report["metric_samples"][key]
        result = f"{value:.4f}" if value is not None else "pending"
        markdown.append(f"| {key} | {result} | {sample['numerator']} | {sample['denominator']} |")
    markdown += ["", "## Human review", "",
                 f"Required: {report['human_review']['required']}; "
                 f"reviewed: {report['human_review']['reviewed']}; "
                 f"pending: {report['human_review']['pending']}", "", "## Acceptance blockers", ""]
    markdown += ["- " + b for b in report["acceptance"]["blockers"]] or ["- None"]
    markdown += ["", "## Failed cases", ""]
    failures = []
    for case in report["cases"]:
        if case["type"] == "ranking":
            for variant, result in case["variants"].items():
                if not result["passed"]:
                    failures.append(f"- {case['id']} / {variant}: "
                                    f"{result.get('error', 'ranking mismatch')}")
                if result.get("human_supported") is False:
                    failures.append(f"- {case['id']} / {variant}: unsupported reason; "
                                    f"{result.get('human_note', '')}")
        elif not case["passed"]:
            failures.append(f"- {case['id']}: {case.get('error', 'gold target not retrieved / extracted')}")
        elif case["type"] == "retrieval" and not case["chunk_passed"]:
            failures.append(f"- {case['id']}: video found but gold subtitle span missed")
    failures += [f"- index / {v['bvid']}: {v['error']}" for v in report["index_errors"]]
    markdown += failures or ["- None"]
    markdown += ["", "## Limitations", ""] + ["- " + item for item in report["limitations"]]
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"json_report": str(path), "markdown_report": str(path.with_suffix(".md")),
            "run_id": report["run_id"], "case_count": report["case_count"],
            "mode": report["mode"], "quality_verified": report["quality_verified"],
            "human_review": report["human_review"], "metrics": report["metrics"]}


def output_fingerprint(result):
    return fingerprint({"reason": result["reason"], "evidence": result["evidence"]})


def review_evaluation(report_path: Path, labels: Path | None = None,
                      output: Path | None = None) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("schema_version") != 2:
        raise ValueError("Review requires a version 2 evaluation report; run eval again")
    entries = {(case["id"], variant): (case, result)
               for case, variant, result in review_outputs(report)}
    if labels is None:
        path = output or report_path.with_name(report_path.stem + ".review.jsonl")
        if path.exists():
            raise ValueError(f"Review template already exists: {path}; use a new --output")
        rows = []
        for (case_id, variant), (case, result) in entries.items():
            profile = result.get("profile") or json.loads(result["retrieval_query"])
            rows.append({"run_id": report["run_id"], "case_id": case_id, "variant": variant,
                         "output_hash": output_fingerprint(result), "profile": profile,
                         "top_bvid": result["top_bvid"], "reason": result["reason"],
                         "evidence": result["evidence"], "supported": None, "note": ""})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        return {"review_template": str(path), "review_count": len(rows), "run_id": report["run_id"]}
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each review line must be a JSON object")
        key = (row.get("case_id"), row.get("variant"))
        if key in seen or key not in entries:
            raise ValueError(f"Duplicate or unknown review target: {key}")
        seen.add(key)
        result = entries[key][1]
        if row.get("run_id") != report["run_id"] or (
            row.get("output_hash") != output_fingerprint(result)
        ):
            raise ValueError(f"Review does not match this run's output: {key}")
        if not isinstance(row.get("supported"), bool) or not isinstance(row.get("note"), str):
            raise ValueError(f"Review supported must be true/false and note must be text: {key}")
    for row in rows:
        result = entries[(row["case_id"], row["variant"])][1]
        result.update(human_supported=row["supported"], human_note=row["note"],
                      human_reviewed_at=utc_now().isoformat())
    report["human_labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
    report["reviewed_at"] = utc_now().isoformat()
    refresh_report(report)
    path = output or report_path.with_name(report_path.stem + ".reviewed.json")
    if path.resolve() == report_path.resolve():
        raise ValueError("Write reviews to a separate report so the original outputs are preserved")
    return write_report(report, path)
