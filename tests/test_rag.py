import hashlib
import json

import pytest

from video_agent.models.contracts import ModelError
from video_agent.rag import IndexMismatch, Rag, split_subtitles
from video_agent.schemas import Segment, Video
from video_agent.store import Store


class Embedding:
    space_id = "fake:v1"

    def embed(self, texts):
        return [[1.0, float(int(hashlib.sha256(t.encode()).hexdigest()[:2], 16))] for t in texts]


class Summarizer:
    def generate_json(self, messages, schema):
        content = json.loads(messages[-1].content)["content"]
        return {"summary": "RAG 实现案例", "evidence_ids": [content[0]["id"]]}


def make_video():
    return Video(bvid="BV0000000001", title="RAG", cid=1, subtitle_status="available",
                 segments=[Segment(start=i, end=i + 1, text="检索增强生成的实现。" * 10)
                           for i in range(15)])


def test_chunks_cover_text_and_keep_source_times():
    video = make_video()
    chunks = split_subtitles(video, Embedding())
    assert len(chunks) > 1
    for segment in video.segments:
        assert any(segment.text in c.text and c.start <= segment.start and c.end >= segment.end
                   for c in chunks)
    assert all(c.bvid == video.bvid and c.cid == 1 for c in chunks)


def test_index_cache_and_embedding_switch(tmp_path):
    store = Store(tmp_path / "agent.sqlite")
    video = make_video()
    store.save_video(video)
    embedding = Embedding()
    rag = Rag(store, Summarizer(), embedding)
    rag.index_video(video)
    first = store.index(video.bvid)
    rag.index_video(video)
    assert store.index(video.bvid) == first
    assert rag.retrieve("RAG", [video])[0]["chunks"]
    embedding.space_id = "fake:v2"
    with pytest.raises(IndexMismatch):
        rag.retrieve("RAG", [video])
    assert rag.reindex() == 1
    assert store.index(video.bvid)["space_id"] == "fake:v2"
    store.close()


def test_invalid_summary_citation_is_rejected(tmp_path):
    class BadSummarizer:
        def generate_json(self, messages, schema):
            return {"summary": "anything", "evidence_ids": ["other-video"]}

    store = Store(tmp_path / "agent.sqlite")
    with pytest.raises(ModelError):
        Rag(store, BadSummarizer(), Embedding()).index_video(make_video())
    assert not store.index_spaces()
    store.close()


def test_chunk_policy_change_rebuilds_cached_index(tmp_path, monkeypatch):
    from video_agent.rag import CHUNK_POLICY

    class CountingEmbedding(Embedding):
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    store = Store(tmp_path / "agent.sqlite")
    try:
        embedding = CountingEmbedding()
        rag = Rag(store, Summarizer(), embedding)
        video = make_video()
        rag.index_video(video)
        first_hash = store.index(video.bvid)["content_hash"]
        calls = embedding.calls
        rag.index_video(video)
        assert embedding.calls == calls
        monkeypatch.setitem(CHUNK_POLICY, "version", "embedding-test-v2")
        rag.index_video(video)
        assert embedding.calls > calls
        assert store.index(video.bvid)["content_hash"] != first_hash
    finally:
        store.close()
