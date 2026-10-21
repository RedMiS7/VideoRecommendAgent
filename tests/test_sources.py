import json

import httpx2
import pytest
from mcp.types import CallToolResult, TextContent

from video_agent.config import Settings
from video_agent.schemas import Video
from video_agent.sources import BilibiliSource, EnhancedSearch, SourceError, video_id


@pytest.mark.parametrize("url", ["https://evil.test/video/BV0000000001",
    "https://bilibili.com.evil.test/video/BV0000000001", "https://www.bilibili.com/read/BV0000000001",
    "https://www.bilibili.com/video/invalid", "http://www.bilibili.com/video/BV0000000001"])
def test_search_url_filter_rejects_nonvideo_and_untrusted_urls(url):
    assert video_id(url) is None


def search_result(pages, **kwargs):
    return CallToolResult(content=[TextContent(type="text", text=json.dumps({
        "status": 0, "pages": pages}))], **kwargs)


def mock_search(monkeypatch, result, settings=None):
    search = EnhancedSearch(settings or Settings(model_api_key="test"))

    async def fetch(query, api_key):
        assert query.startswith("site:bilibili.com ") and len(query) <= 500
        assert api_key == "test"
        return result

    monkeypatch.setattr(search, "_fetch", fetch)
    return search


def test_search_normalizes_duplicates_and_html_titles(monkeypatch):
    hits = [{"url": "https://www.bilibili.com/video/BV0000000001/?p=2", "title": "<em>RAG</em>",
             "snippet": "test"}, {"url": "https://evil.test/video/BV0000000001", "title": "fake"},
            {"url": "https://www.bilibili.com/video/BV0000000001", "title": "duplicate"}]
    search = mock_search(monkeypatch, search_result(hits))
    result = search.search("RAG")
    assert len(result) == 1 and result[0].title == "RAG"
    assert result[0].url.endswith("?p=1")
    assert result[0].search_summary == "test"
    search.close()


def test_search_empty_pages_and_structured_output(monkeypatch):
    search = mock_search(monkeypatch, CallToolResult(content=[],
        structured_content={"status": 0, "pages": []}))
    assert search.search("RAG") == []


def test_mobile_bilibili_video_is_canonicalized_but_other_mobile_hosts_are_rejected(monkeypatch):
    search = mock_search(monkeypatch, search_result([
        {"url": "https://m.bilibili.com/video/BV0000000001/?p=2", "title": "RAG"},
        {"url": "https://m.bilibili.com.evil.test/video/BV0000000002", "title": "fake"},
        {"url": "https://m.bilibili.com/space/123", "title": "channel"},
    ]))
    result = search.search("RAG")
    assert [v.url for v in result] == ["https://www.bilibili.com/video/BV0000000001/?p=1"]


@pytest.mark.parametrize("result,kind", [
    (CallToolResult(content=[TextContent(type="text", text="private server error")],
                    is_error=True), "unavailable"),
    (CallToolResult(content=[TextContent(type="text", text="not JSON")]), "invalid_output"),
    (CallToolResult(content=[TextContent(type="text", text='{"status": 2}')]), "unavailable"),
    (CallToolResult(content=[TextContent(type="text", text='{"pages": null}')]), "invalid_output"),
    (search_result([{"url": 123}]), "invalid_output"),
])
def test_search_reports_tool_errors_and_malformed_results(monkeypatch, result, kind):
    search = mock_search(monkeypatch, result)
    with pytest.raises(SourceError) as error:
        search.search("RAG")
    assert error.value.kind == kind
    assert "private" not in str(error.value)


def test_search_requires_key_before_connecting(monkeypatch):
    search = EnhancedSearch(Settings())
    with pytest.raises(SourceError, match="SEARCH_API_KEY or DASHSCOPE_API_KEY") as error:
        search.search("RAG")
    assert error.value.kind == "authentication"


def test_search_has_independent_key_and_limits_query(monkeypatch):
    search = mock_search(monkeypatch, search_result([]),
                         Settings(search_api_key="test", model_api_key="other"))
    assert search.search("x" * 600) == []


def test_search_performs_mcp_handshake_and_tool_call(monkeypatch):
    from contextlib import asynccontextmanager
    from video_agent import sources

    events = []

    @asynccontextmanager
    async def transport(url, *, http_client):
        assert url == Settings.search_mcp_url
        assert http_client.headers["Authorization"] == "Bearer test"
        yield ("reader", "writer", None)
        events.append("transport_closed")

    class Session:
        def __init__(self, read, write, **kwargs):
            assert (read, write) == ("reader", "writer")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            events.append("session_closed")

        async def initialize(self):
            events.append("initialize")

        async def call_tool(self, name, arguments):
            events.append("call_tool")
            assert name == "search_pro"
            assert arguments == {"query": "site:bilibili.com RAG"}
            return search_result([])

    monkeypatch.setattr(sources, "streamable_http_client", transport)
    monkeypatch.setattr(sources, "ClientSession", Session)
    assert EnhancedSearch(Settings(model_api_key="test")).search("RAG") == []
    assert events == ["initialize", "call_tool", "session_closed", "transport_closed"]


def test_search_sanitizes_transport_failure(monkeypatch):
    search = EnhancedSearch(Settings(model_api_key="test"))

    async def fail(*args):
        request = httpx2.Request("POST", "https://search.test/mcp")
        response = httpx2.Response(401, request=request)
        raise httpx2.HTTPStatusError("secret credential body", request=request, response=response)

    monkeypatch.setattr(search, "_fetch", fail)
    with pytest.raises(SourceError) as error:
        search.search("RAG")
    assert error.value.kind == "authentication"
    assert "secret" not in str(error.value)


def test_first_part_metadata_and_missing_vs_login_subtitles(monkeypatch):
    from bilibili_api import video

    class FakeVideo:
        track_info = {"login_mid": 1, "subtitle": {"subtitles": []}}

        def __init__(self, **kwargs):
            pass

        async def get_info(self):
            return {"title": "course", "duration": 2000,
                    "pages": [{"cid": 1, "duration": 500, "part": "First"},
                              {"cid": 2, "duration": 1500, "part": "Second"}],
                    "stat": {"view": 100, "like": 20, "favorite": 10, "reply": 3}}

        async def get_player_info(self, cid):
            assert cid == 1
            return self.track_info

    monkeypatch.setattr(video, "Video", FakeVideo)
    source = BilibiliSource(Settings(bili_sessdata="test"))
    try:
        result = source.metadata(Video(bvid="BV0000000001", title="search"))
        assert result.cid == 1 and result.part_count == 2 and result.duration == 500
        absent = source.subtitles(result)
        assert absent.subtitle_status == "absent"
        FakeVideo.track_info = {"login_mid": 0, "need_login_subtitle": True,
                                "subtitle": {"subtitles": []}}
        login = source.subtitles(result)
        assert login.subtitle_status == "error" and "login" in login.subtitle_error
    finally:
        source.close()


def test_subtitle_missing_sessdata_reports_configuration_before_sdk_call(monkeypatch):
    from bilibili_api import video

    def unexpected_sdk_call(**kwargs):
        pytest.fail("Subtitle SDK should not be called without SESSDATA")

    monkeypatch.setattr(video, "Video", unexpected_sdk_call)
    source = BilibiliSource(Settings())
    try:
        result = source.subtitles(Video(bvid="BV0000000001", title="RAG", cid=1))
        assert result.subtitle_status == "error"
        assert "BILI_SESSDATA" in result.subtitle_error
        assert result.segments == [] and result.subtitle_checked_at is not None
    finally:
        source.close()


def test_bilibili_http_error_reports_status_without_response_body():
    from bilibili_api.exceptions import NetworkException

    async def fail():
        raise NetworkException(412, "private response body")

    source = BilibiliSource(Settings())
    try:
        with pytest.raises(SourceError, match="HTTP 412") as error:
            source._run(fail())
        assert "private" not in str(error.value)
    finally:
        source.close()


def test_subtitle_error_is_retried_after_credentials_are_configured(monkeypatch):
    from bilibili_api import video
    from video_agent.schemas import utc_now

    calls = []

    class FakeVideo:
        def __init__(self, **kwargs):
            pass

        async def get_player_info(self, cid):
            calls.append(cid)
            return {"login_mid": 1, "subtitle": {"subtitles": []}}

    monkeypatch.setattr(video, "Video", FakeVideo)
    source = BilibiliSource(Settings(bili_sessdata="test"))
    try:
        cached_error = Video(bvid="BV0000000001", title="RAG", cid=1,
            subtitle_status="error", subtitle_error="Missing credentials",
            subtitle_checked_at=utc_now())
        result = source.subtitles(cached_error)
        assert calls == [1]
        assert result.subtitle_status == "absent" and result.subtitle_error is None
        assert source.subtitles(result) == result
        assert calls == [1, 1]  # Empty lists are rechecked to recover old login misclassifications.
    finally:
        source.close()


def test_old_absent_cache_is_rechecked_and_ai_chinese_is_downloaded(monkeypatch):
    from bilibili_api import video
    from video_agent.schemas import utc_now

    class FakeVideo:
        def __init__(self, **kwargs):
            pass

        async def get_player_info(self, cid):
            assert cid == 1
            return {"login_mid": 1, "subtitle": {"subtitles": [
                {"lan": "en", "subtitle_url": "https://i0.hdslb.com/english.json"},
                {"lan": "ai-zh", "subtitle_url": "//i0.hdslb.com/chinese.json"},
            ]}}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aclose(self):
            pass

        async def get(self, url):
            import httpx
            assert url == "https://i0.hdslb.com/chinese.json"
            return httpx.Response(200, request=httpx.Request("GET", url),
                json={"body": [{"from": 0, "to": 2, "content": "AI 中文字幕"}]})

    monkeypatch.setattr(video, "Video", FakeVideo)
    monkeypatch.setattr("video_agent.sources.httpx.AsyncClient", Client)
    source = BilibiliSource(Settings(bili_sessdata="test"))
    try:
        cached = Video(bvid="BV0000000001", title="RAG", cid=1, subtitle_status="absent",
                       subtitle_checked_at=utc_now())
        result = source.subtitles(cached)
        assert result.subtitle_status == "available"
        assert result.segments[0].text == "AI 中文字幕"
        assert source.subtitles(result) == result
    finally:
        source.close()
