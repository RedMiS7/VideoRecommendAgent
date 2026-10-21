import hashlib
import json
import math
import time
import uuid
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .models.contracts import ModelError
from .rag import CHUNK_POLICY, IndexMismatch, cosine
from .runtime import Runtime
from .schemas import Profile, utc_now


class _FrozenSource:
    def __init__(self, videos):
        self.videos = videos

    def search(self, query):
        return [video.model_copy(deep=True) for video in self.videos]

    def metadata(self, video):
        return video

    def subtitles(self, video):
        return video


def recommendation_metrics(items, relevance, vectors):
    ids = [item["bvid"] for item in items[:5]]
    if len(ids) != len(set(ids)) or any(bvid not in relevance for bvid in ids):
        raise ValueError("Recommendations must be unique and belong to the labeled corpus")
    precision, ndcg = None, None
    if None not in relevance.values():
        grades = [relevance[bvid] for bvid in ids]
        precision = sum(grade >= 2 for grade in grades) / 5
        dcg = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
        ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(
            sorted(relevance.values(), reverse=True)[:5]))
        ndcg = dcg / ideal if ideal else None
    diversity = None
    if len(ids) >= 2 and all(bvid in vectors and np.linalg.norm(vectors[bvid]) > 0 for bvid in ids):
        distances = [1 - max(-1.0, min(1.0, cosine(vectors[a], vectors[b])))
                     for a, b in combinations(ids, 2)]
        diversity = sum(distances) / len(distances)
    return {"precision_at_5": precision, "ndcg_at_5": ndcg, "diversity_at_5": diversity}


def _load_pricing(path):
    if path is None:
        return None
    pricing = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(pricing.get("currency"), str) or not pricing["currency"].strip() or (
        not isinstance(pricing.get("models"), dict)
    ):
        raise ValueError("Pricing requires currency and a models rate table")
    for rates in pricing["models"].values():
        if not isinstance(rates, dict):
            raise ValueError("Each model needs input/output rates per million tokens")
        for name in ("input_per_million", "output_per_million"):
            rate = rates.get(name)
            if rate is not None and (type(rate) not in (int, float) or
                                     not math.isfinite(rate) or rate < 0):
                raise ValueError("Token rates must be nonnegative finite numbers or null")
    return pricing


def usage_cost(calls, pricing):
    input_tokens, output_tokens, cost = 0, 0, 0.0
    missing_usage, missing_rates = 0, 0
    for call in calls:
        usage = call.get("usage") or {}
        is_embedding = call.get("operation") == "/embeddings"
        incoming = usage.get("prompt_tokens", usage.get("input_tokens"))
        outgoing = 0 if is_embedding else usage.get("completion_tokens", usage.get("output_tokens"))
        if is_embedding and incoming is None:
            incoming = usage.get("total_tokens")
        if any(type(value) is not int or value < 0 for value in (incoming, outgoing)):
            missing_usage += 1
            continue
        input_tokens += incoming
        output_tokens += outgoing
        rates = (pricing or {}).get("models", {}).get(call["model"], {})
        input_rate, output_rate = rates.get("input_per_million"), rates.get("output_per_million")
        if input_rate is None or (outgoing and output_rate is None):
            missing_rates += 1
        else:
            cost += (incoming * input_rate + outgoing * (output_rate or 0)) / 1_000_000
    return {"model_requests": len(calls),
            "input_tokens": input_tokens if not missing_usage else None,
            "output_tokens": output_tokens if not missing_usage else None,
            "total_tokens": input_tokens + output_tokens if not missing_usage else None,
            "estimated_model_cost": cost if not missing_usage and not missing_rates else None,
            "missing_usage_requests": missing_usage, "missing_rate_requests": missing_rates}


def _aggregate(records):
    metrics, samples = {}, {}
    for name in ("precision_at_5", "ndcg_at_5", "diversity_at_5"):
        values = [record["metrics"][name] for record in records
                  if record["metrics"][name] is not None]
        metrics[name] = sum(values) / len(values) if values else None
        samples[name] = {"valid": len(values), "total": len(records)}
    costs = [record["cost"]["estimated_model_cost"] for record in records]
    tokens = [record["cost"]["total_tokens"] for record in records]
    durations = [record["seconds"] for record in records]
    metrics["cost_latency"] = {
        "mean_seconds": sum(durations) / len(durations),
        "p50_seconds": float(np.percentile(durations, 50)),
        "p95_seconds": float(np.percentile(durations, 95)),
        "mean_total_tokens": sum(tokens) / len(tokens) if None not in tokens else None,
        "mean_estimated_model_cost": sum(costs) / len(costs) if None not in costs else None,
        "estimated_model_cost_total": sum(costs) if None not in costs else None}
    samples["cost_latency"] = {"priced_cases": sum(cost is not None for cost in costs),
                               "total": len(records)}
    return metrics, samples


def _write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"eval-{report['run_id']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def display(value):
        return f"{value:.4f}" if value is not None else "pending / undefined"

    pending = report["label_review"]["pending"]
    lines = ["# Video Agent Recommendation Evaluation", "", f"Run: {report['run_id']}",
             f"Mode: {report['mode']}", f"Cases: {report['case_count']}",
             f"Human label review pending: {pending}",
             "Scores using draft labels are provisional. No automatic quality threshold is set.",
             "", "| Metric | Result | Valid cases / Total |", "| --- | --- | --- |"]
    if report["mode"] == "offline_test_doubles":
        lines.insert(6, "**Offline scores and timings only verify the pipeline; they do not measure model quality or service performance.**")
    for name in ("precision_at_5", "ndcg_at_5", "diversity_at_5"):
        sample = report["metric_samples"][name]
        lines.append(f"| {name} | {display(report['metrics'][name])} | "
                     f"{sample['valid']} / {sample['total']} |")
    costs = report["metrics"]["cost_latency"]
    lines += ["", "## Recommendation cost and latency", "",
              f"Currency: {report['currency'] or 'not configured'}", "",
              "| Measure | Result |", "| --- | --- |"]
    lines += [f"| {key} | {display(value)} |" for key, value in costs.items()]
    lines += ["", "## Per-profile results", "",
              "| Case | Status | Returned | Precision@5 | NDCG@5 | Diversity@5 | Seconds |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for record in report["cases"]:
        scores = record["metrics"]
        lines.append(f"| {record['id']} | {record['status']} | {len(record['items'])} | "
                     f"{display(scores['precision_at_5'])} | {display(scores['ndcg_at_5'])} | "
                     f"{display(scores['diversity_at_5'])} | {record['seconds']:.3f} |")
    lines += ["", "## Failures", ""]
    lines += [f"- {r['id']}: {r['error']}" for r in report["cases"] if r["status"] == "error"] \
        or ["- None"]
    lines += ["", "## Limitations", ""] + ["- " + item for item in report["limitations"]]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_report": str(path), "markdown_report": str(path.with_suffix(".md")),
            "run_id": report["run_id"], "mode": report["mode"], "metrics": report["metrics"],
            "label_review": report["label_review"]}


def run_recommendation_evaluation(settings, dataset: Path, output: Path, cases, videos,
                                  manifest, offline=False, pricing_path=None):
    from .evaluation import _TracedLanguageModel

    pricing = _load_pricing(pricing_path)
    session = uuid.uuid4().hex[:12]
    records, prompts = [], {}
    started = time.monotonic()
    for position, case in enumerate(cases):
        runtime = Runtime(replace(settings, data_dir=settings.data_dir / "eval" / session /
                                  str(position)), offline=offline)
        try:
            traced = _TracedLanguageModel(runtime.models.language, prompts)
            runtime.models.language = runtime.rag.model = runtime.memory.model = traced
            runtime.recommender.model = traced
            frozen = _FrozenSource(videos)
            runtime.recommender.search = runtime.recommender.source = frozen
            runtime.store.save_profile(Profile.model_validate(case["profile"]))
            record = {**case, "status": "success", "items": [], "warnings": []}
            start = time.monotonic()
            try:
                result = runtime.recommender.run()
                record.update(items=result["items"][:5], warnings=result["warnings"])
            except (ModelError, IndexMismatch, ValueError) as error:
                record.update(status="error", error={
                    "kind": getattr(error, "kind", type(error).__name__), "message": str(error)})
            record["seconds"] = round(time.monotonic() - start, 6)
            record["model_calls"] = list(runtime.models.calls.events)
            record["cost"] = usage_cost(record["model_calls"], pricing)
            if offline:
                record["cost"].update(model_requests=None, input_tokens=None, output_tokens=None,
                                      total_tokens=None, estimated_model_cost=None)
            vectors = {item["bvid"]: index["vector"] for item in record["items"]
                       if (index := runtime.store.index(item["bvid"]))}
            record["diversity_vectors"] = vectors
            record["diversity_missing_bvids"] = [item["bvid"] for item in record["items"]
                                                if item["bvid"] not in vectors]
            record["metrics"] = recommendation_metrics(record["items"], case["relevance"], vectors)
            records.append(record)
        finally:
            runtime.close()
    metrics, samples = _aggregate(records)
    reviewed = sum(case["label_status"] == "human_reviewed" for case in cases)
    mode = "offline_test_doubles" if offline else (
        "real_models_real_dataset" if manifest.get("source") == "bilibili" else
        "real_models_synthetic_dataset" if manifest.get("source") == "synthetic" else
        "real_models_unspecified_dataset")
    report = {"schema_version": 3, "suite": "recommendation_at_5", "run_id": session,
              "created_at": utc_now().isoformat(), "mode": mode,
              "dataset": str(dataset.resolve()), "dataset_manifest": manifest,
              "case_count": len(cases), "video_count": len(videos),
              "label_review": {"required": len(cases), "reviewed": reviewed,
                               "pending": len(cases) - reviewed},
              "metrics": metrics, "metric_samples": samples,
              "metrics_by_split": {split: _aggregate([r for r in records if r["split"] == split])[0]
                                   for split in sorted({r["split"] for r in records})},
              "metric_config": {"k": 5, "precision_relevance_threshold": 2,
                                "ndcg_gain": "2**grade - 1", "ndcg_discount": "log2(rank + 1)",
                                "diversity": "mean pairwise subtitle-summary cosine distance",
                                "cost_latency_scope": "frozen corpus, independent cold cache"},
              "models": {"text": settings.text_model, "embedding": settings.embedding_model}
              if not offline else {"all": "deterministic_test_doubles"},
              "prompt_hashes": {key: sorted(values) for key, values in prompts.items()},
              "chunk_policy": dict(CHUNK_POLICY), "pricing": pricing,
              "pricing_sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest()
              if pricing_path else None,
              "currency": pricing["currency"] if pricing else None,
              "seconds": round(time.monotonic() - started, 3), "cases": records,
              "limitations": [
                  "Offline test-double scores and timings do not measure AI quality or real-service performance",
                  "Draft relevance labels are provisional; human review is required",
                  "Frozen candidates replace live discovery and subtitle downloads; network latency and search fees are excluded",
                  "Each profile starts with an empty cache; model costs and latency include first-time indexing, unlike a warm daily run",
                  "Precision uses five slots even for short lists; small pools with few relevant videos limit the attainable score",
                  "NDCG is undefined when all candidate grades are zero; null grades leave Precision and NDCG pending",
                  "Diversity is cosine distance in [0, 2], not a percentage; fewer than two items or any missing subtitle-summary vector makes it undefined",
                  "High diversity alone does not establish relevance; read it alongside Precision and NDCG",
                  "Money estimates use supplied flat token rates; missing usage/rates leave the total unknown, and provider billing discounts are not inferred",
                  "No automatic pass thresholds are configured; results apply only to this frozen dataset"
              ] + manifest.get("limitations", [])}
    return _write_report(report, output)
