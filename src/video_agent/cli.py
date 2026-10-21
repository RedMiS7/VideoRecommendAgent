import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .models.contracts import Message, ModelError
from .rag import IndexMismatch
from .runtime import Runtime
from .schemas import FeedbackInput, Profile, Video
from .sources import SourceError
from .store import RunBusy


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def save_profile(runtime, path):
    profile = Profile.model_validate_json(Path(path).read_text(encoding="utf-8-sig"))
    for preference in profile.preferences:
        preference.source = "explicit"
        preference.evidence_ids = []
    runtime.store.save_profile(profile)
    return profile.model_dump()


def parser():
    root = argparse.ArgumentParser(description="Local learning-video recommender")
    root.add_argument("--data-dir", type=Path)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Initialize from a file, or generate a text draft")
    source = init.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--text")
    profile = commands.add_parser("profile", help="View or replace explicit profile settings")
    profile.add_argument("--file", type=Path)
    feedback = commands.add_parser("feedback")
    feedback.add_argument("bvid")
    feedback.add_argument("text")
    recommend = commands.add_parser("recommend", help="Run today, or query a saved date")
    recommend.add_argument("--date", help="Query YYYY-MM-DD; does not generate a past run")
    commands.add_parser("memory")
    commands.add_parser("reindex")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    check = commands.add_parser("live-check", help="Probe real data and configured model/search services")
    check.add_argument("--bvid", action="append", required=True)
    demo = commands.add_parser("demo", help="Explicit synthetic offline demonstration")
    demo.add_argument("--days", type=int, choices=range(1, 9), default=1)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--offline", action="store_true")
    evaluation.add_argument("--dataset", type=Path, default=Path("evals/recommendation-cases.jsonl"))
    evaluation.add_argument("--output", type=Path, default=Path("reports"))
    evaluation.add_argument("--split", choices=("dev", "test"))
    evaluation.add_argument("--pricing", type=Path, help="Actual model token rates per million")
    review = commands.add_parser("eval-review", help="Export or import human evidence reviews")
    review.add_argument("--report", type=Path, required=True)
    review.add_argument("--labels", type=Path)
    review.add_argument("--output", type=Path)
    collect = commands.add_parser("eval-collect", help="Freeze Bilibili videos for evaluation")
    collect.add_argument("--bvid-file", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    return root


def live_check(runtime, bvids):
    checks = []
    if len(bvids) > 3:
        raise ValueError("Probe at most three public videos per run")
    for bvid in bvids:
        try:
            video = runtime.source.metadata(Video(bvid=bvid, title="Probe"))
            video = runtime.source.subtitles(video)
            checks.append({"service": "bilibili", "bvid": bvid, "metadata": "success",
                           "title": video.title, "part_count": video.part_count,
                           "stats": video.stats.model_dump(), "subtitle_status": video.subtitle_status,
                           "subtitle_error": video.subtitle_error, "segments": len(video.segments)})
        except SourceError as error:
            checks.append({"service": "bilibili", "bvid": bvid, "status": "failed",
                           "kind": error.kind, "error": str(error)})
    for name, function in [
        ("language", lambda: runtime.models.language.generate_json(
            [Message("user", 'Return {"ok": true}')], {"type": "object", "properties": {
                "ok": {"const": True}}, "required": ["ok"], "additionalProperties": False})),
        ("embedding", lambda: len(runtime.models.embedding.embed(["RAG 学习视频"])[0])),
        ("search", lambda: len(runtime.search.search("RAG 学习视频"))),
    ]:
        try:
            checks.append({"service": name, "status": "success", "result": function()})
        except (ModelError, SourceError) as error:
            checks.append({"service": name, "status": "failed", "kind": error.kind,
                           "error": str(error)})
    result = {"mode": "real_service_probe", "checks": checks}
    runtime.settings.data_dir.mkdir(parents=True, exist_ok=True)
    (runtime.settings.data_dir / "live-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    args = parser().parse_args()
    settings = Settings.from_env()
    if args.data_dir:
        settings = replace(settings, data_dir=args.data_dir.resolve())
    try:
        if args.command == "eval":
            from .evaluation import run_evaluation

            output(run_evaluation(settings, args.dataset, args.output, args.offline,
                                  args.split, args.pricing))
            return
        if args.command == "eval-review":
            from .evaluation_reports import review_evaluation

            output(review_evaluation(args.report, args.labels, args.output))
            return
        if args.command == "eval-collect":
            from .eval_corpus import collect_corpus

            result = collect_corpus(settings, args.bvid_file, args.output)
            output(result)
            if not result["complete"]:
                raise SystemExit(1)
            return
        if args.command == "serve":
            import uvicorn
            from .api import create_app

            runtime = Runtime(settings)
            try:
                uvicorn.run(create_app(runtime), host="127.0.0.1", port=args.port)
            finally:
                runtime.close()
            return
        offline = args.command == "demo"
        if offline:
            settings = replace(settings, data_dir=settings.data_dir / "offline-demo")
        runtime = Runtime(settings, offline=offline)
        try:
            if args.command == "init":
                if args.file:
                    output(save_profile(runtime, args.file))
                else:
                    profile = runtime.memory.initialize_draft(args.text)
                    path = settings.data_dir / "profile-draft.json"
                    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
                    output({"draft": profile.model_dump(), "saved_to": str(path),
                            "next": f"Review this draft, then run: video-agent init --file {path}"})
            elif args.command == "profile":
                output(save_profile(runtime, args.file) if args.file else (
                    runtime.store.profile().model_dump() if runtime.store.profile() else None))
            elif args.command == "memory":
                output({"profile": runtime.store.profile().model_dump() if runtime.store.profile() else None,
                        "feedback": runtime.store.feedback(), "clicks": runtime.store.clicks()})
            elif args.command == "feedback":
                output(runtime.memory.submit(FeedbackInput(bvid=args.bvid, text=args.text)))
            elif args.command == "recommend":
                if args.date:
                    from datetime import date

                    date.fromisoformat(args.date)
                    output(runtime.store.run(args.date))
                else:
                    output(runtime.recommender.run())
            elif args.command == "reindex":
                output({"reindexed": runtime.rag.reindex()})
            elif args.command == "live-check":
                result = live_check(runtime, args.bvid)
                output(result)
                if any(check.get("status") == "failed" or check.get("subtitle_status") == "error"
                       for check in result["checks"]):
                    raise SystemExit(1)
            elif args.command == "demo":
                from datetime import timedelta
                from .schemas import utc_now

                if not runtime.store.profile():
                    runtime.store.save_profile(Profile(topics=["RAG"], goal="学习工程实践"))
                now = utc_now()
                output({"mode": "synthetic_offline_demo", "warning":
                        "Uses test doubles and synthetic videos; not real model or Bilibili results",
                        "days": [runtime.recommender.run(now + timedelta(days=i))
                                 for i in range(args.days)]})
        finally:
            runtime.close()
    except (ModelError, SourceError, IndexMismatch, RunBusy, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
