import asyncio
import html
import json
import re
import threading
from datetime import timedelta
from urllib.parse import urlparse

import httpx
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import Settings
from .schemas import Segment, Video, VideoStats, utc_now


class SourceError(RuntimeError):
    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


def video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "www.bilibili.com", "bilibili.com", "m.bilibili.com"
    }:
        return None
    match = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10})/?", parsed.path)
    return match.group(1) if match else None


class EnhancedSearch:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def _fetch(self, query: str, api_key: str):
        async with httpx2.AsyncClient(timeout=30,
            headers={"Authorization": "Bearer " + api_key}) as client:
            async with streamable_http_client(self.settings.search_mcp_url,
                                               http_client=client) as streams:
                async with ClientSession(streams[0], streams[1], read_timeout_seconds=30) as session:
                    await session.initialize()
                    return await session.call_tool("search_pro", {"query": query})

    def search(self, query: str) -> list[Video]:
        api_key = self.settings.search_api_key or self.settings.model_api_key
        if not api_key:
            raise SourceError("authentication", "SEARCH_API_KEY or DASHSCOPE_API_KEY is required")
        prefix = "site:bilibili.com "
        query = prefix + query.strip()[:500 - len(prefix)]
        try:
            response = asyncio.run(asyncio.wait_for(self._fetch(query, api_key), timeout=45))
        except Exception as error:
            # MCP transport errors can be nested in task-group exceptions; never log their bodies.
            pending, kind = [error], "unavailable"
            while pending:
                current = pending.pop()
                if isinstance(current, httpx2.HTTPStatusError):
                    if current.response.status_code in (401, 403):
                        kind = "authentication"
                pending.extend(getattr(current, "exceptions", ()))
            raise SourceError(kind, "EnhancedSearch MCP request failed") from None
        try:
            if response.is_error:
                raise SourceError("unavailable", "EnhancedSearch tool returned an error")
            data = response.structured_content
            if data is None:
                data = json.loads("\n".join(block.text for block in response.content
                                           if block.type == "text"))
            if not isinstance(data, dict):
                raise ValueError("Expected a search result object")
            if data.get("status", 0) != 0:
                raise SourceError("unavailable", "Search service returned an error")
            hits = data["pages"]
            if not isinstance(hits, list):
                raise ValueError("Expected search pages")
            result, seen = [], set()
            for hit in hits:
                if not isinstance(hit, dict) or not isinstance(hit.get("url"), str):
                    raise ValueError("Expected a page URL")
                bvid = video_id(hit["url"])
                if bvid and bvid not in seen:
                    result.append(Video(bvid=bvid, title=html.unescape(
                        re.sub(r"<[^>]+>", "", hit.get("title") or "")),
                        search_summary=hit.get("snippet") or ""))
                    seen.add(bvid)
            return result
        except (ValueError, KeyError, TypeError):
            raise SourceError("invalid_output", "Invalid EnhancedSearch response") from None

    def close(self):
        # Each search closes its own MCP session and HTTP client.
        pass


class BilibiliSource:
    def __init__(self, settings: Settings):
        from bilibili_api import Credential, select_client

        select_client("httpx")
        self.credential = Credential(sessdata=settings.bili_sessdata or None,
                                     bili_jct=settings.bili_jct or None,
                                     buvid3=settings.bili_buvid3 or None)
        self.loop = asyncio.new_event_loop()
        self.lock = threading.Lock()
        self.started = False

    def _run(self, coroutine):
        with self.lock:
            self.started = True
            try:
                return self.loop.run_until_complete(asyncio.wait_for(coroutine, timeout=45))
            except Exception as error:
                # SDK exceptions are normalized here; credentials/response bodies are never logged.
                code = getattr(error, "code", None)
                kind = "not_found" if code in (-404, 62002, 62004) else "unavailable"
                status = getattr(error, "status", None)
                detail = f", HTTP {status}" if isinstance(status, int) else ""
                raise SourceError(kind,
                    f"Bilibili request failed ({type(error).__name__}{detail})") from None

    def metadata(self, item: Video) -> Video:
        if item.metadata_at and utc_now() - item.metadata_at < timedelta(hours=24):
            return item

        async def fetch():
            from bilibili_api import video

            info = await video.Video(bvid=item.bvid, credential=self.credential).get_info()
            pages = info.get("pages") or []
            if not pages:
                raise ValueError("Video has no parts")
            part = pages[0]
            stats = info.get("stat") or {}
            result = Video.model_validate({**item.model_dump(),
                "title": info["title"], "description": info.get("desc", ""),
                "uploader": (info.get("owner") or {}).get("name", ""),
                "duration": part.get("duration"), "published_at": info.get("pubdate"),
                "cid": part["cid"], "part_count": len(pages), "part_title": part.get("part", ""),
                "stats": VideoStats(views=stats.get("view"), likes=stats.get("like"),
                                    favorites=stats.get("favorite"), comments=stats.get("reply")),
                "metadata_at": utc_now(),
            })
            if item.cid and item.cid != result.cid:
                result.segments = []
                result.subtitle_status = "unknown"
                result.subtitle_checked_at = None
            return result

        return self._run(fetch())

    def subtitles(self, item: Video) -> Video:
        if (item.subtitle_status == "available" and item.subtitle_checked_at and
            utc_now() - item.subtitle_checked_at < timedelta(hours=24)):
            return item
        if not item.cid:
            return item.model_copy(update={"subtitle_status": "error",
                "subtitle_error": "Metadata unavailable; CID is unknown",
                "subtitle_checked_at": utc_now()})
        if not self.credential.has_sessdata():
            return item.model_copy(update={"subtitle_status": "error", "segments": [],
                "subtitle_error": "BILI_SESSDATA is required for subtitles; configure it in .env "
                                  "and restart the service",
                "subtitle_checked_at": utc_now()})

        async def fetch():
            from bilibili_api import video

            player = await video.Video(bvid=item.bvid, credential=self.credential).get_player_info(
                cid=item.cid)
            if player.get("need_login_subtitle") or not player.get("login_mid"):
                return item.model_copy(update={"subtitle_status": "error", "segments": [],
                    "subtitle_error": "Subtitle access requires a valid Bilibili login; "
                                      "refresh BILI_SESSDATA in .env and restart the service",
                    "subtitle_checked_at": utc_now()})
            info = player.get("subtitle")
            if not isinstance(info, dict) or not isinstance(info.get("subtitles"), list):
                raise ValueError("Invalid subtitle response")
            tracks = info["subtitles"]
            if not tracks:
                return item.model_copy(update={"subtitle_status": "absent", "segments": [],
                    "subtitle_error": None, "subtitle_checked_at": utc_now()})
            track = next((t for t in tracks if str(t.get("lan", "")).startswith(
                ("zh", "ai-zh"))), tracks[0])
            url = track["subtitle_url"]
            if url.startswith("//"):
                url = "https:" + url
            url = url.replace("http://", "https://", 1)
            host = urlparse(url).hostname or ""
            if urlparse(url).scheme != "https" or not any(
                host == domain or host.endswith("." + domain)
                for domain in ("hdslb.com", "bilibili.com")
            ):
                raise ValueError("Unexpected subtitle host")
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url)
                response.raise_for_status()
                segments = [Segment(start=line["from"], end=line["to"], text=line["content"])
                            for line in response.json()["body"] if line.get("content", "").strip()]
            if not segments:
                raise ValueError("Empty subtitle track")
            return item.model_copy(update={"segments": segments, "subtitle_status": "available",
                "subtitle_error": None, "subtitle_checked_at": utc_now()})

        try:
            return self._run(fetch())
        except SourceError as error:
            return item.model_copy(update={"subtitle_status": "error", "segments": [],
                "subtitle_error": str(error), "subtitle_checked_at": utc_now()})

    def close(self):
        from bilibili_api import get_client

        async def close_client():
            await get_client().close()

        with self.lock:
            if self.started:
                self.loop.run_until_complete(close_client())
            self.loop.close()
