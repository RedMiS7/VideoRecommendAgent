from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import Field

from .config import Settings
from .models.contracts import ModelError
from .rag import IndexMismatch
from .recommender import TIMEZONE
from .runtime import Runtime
from .schemas import FeedbackInput, Profile, Record, utc_now
from .sources import SourceError
from .store import RunBusy


class DraftInput(Record):
    text: str = Field(min_length=1, max_length=4000)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    owns_runtime = runtime is None
    runtime = runtime or Runtime(Settings.from_env())

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owns_runtime:
            runtime.close()

    app = FastAPI(title="Video Agent MVP", lifespan=lifespan)

    @app.exception_handler(ModelError)
    @app.exception_handler(SourceError)
    async def upstream_error(request, error):
        return JSONResponse(status_code=503, content={"detail": str(error), "kind": error.kind})

    @app.exception_handler(IndexMismatch)
    @app.exception_handler(RunBusy)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.get("/health")
    def health():
        return {"status": "ok", "profile_initialized": runtime.store.profile() is not None,
                "mode": "offline_test_doubles" if runtime.offline else "live"}

    @app.get("/profile")
    def profile():
        result = runtime.store.profile()
        if not result:
            raise HTTPException(404, "Profile is not initialized")
        return result

    @app.put("/profile")
    def save_profile(profile: Profile):
        for preference in profile.preferences:
            preference.source = "explicit"
            preference.evidence_ids = []
        runtime.store.save_profile(profile)
        return profile

    @app.post("/profile/draft")
    def draft(input: DraftInput):
        return runtime.memory.initialize_draft(input.text)

    @app.get("/memory")
    def memory():
        return {"profile": runtime.store.profile(), "feedback": runtime.store.feedback(),
                "clicks": runtime.store.clicks()}

    @app.post("/feedback")
    def feedback(input: FeedbackInput):
        try:
            return runtime.memory.submit(input)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    def with_links(payload, request):
        if not payload:
            return payload
        return {**payload, "mode": "offline_test_doubles" if runtime.offline else "live",
                "items": [{**item, "click_url": str(request.url_for("click", bvid=item["bvid"]))}
                          for item in payload["items"]]}

    @app.post("/recommendations/run")
    def run(request: Request):
        try:
            return with_links(runtime.recommender.run(), request)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/recommendations")
    def recommendations(request: Request, date: date | None = None):
        day = (date or utc_now().astimezone(TIMEZONE).date()).isoformat()
        record = runtime.store.run(day)
        if not record:
            raise HTTPException(404, "No recommendation run for this date")
        return {**record, "payload": with_links(record["payload"], request)}

    @app.get("/click/{bvid}", name="click")
    def click(bvid: str):
        video = runtime.store.video(bvid)
        if not video:
            raise HTTPException(404, "Unknown video")
        if runtime.offline:
            raise HTTPException(409, "Synthetic offline videos cannot be opened on Bilibili")
        runtime.store.add_click(bvid)
        return RedirectResponse(video.url, status_code=302)

    return app
