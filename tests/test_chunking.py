import numpy as np
import pytest

from video_agent.models.contracts import ModelError
from video_agent.rag import split_subtitles
from video_agent.schemas import Segment, Video


class TopicEmbedding:
    def embed(self, texts):
        return [[text.count("甲"), text.count("乙") + text.count("丙") * 0.1, 1.0]
                for text in texts]


def video_with(texts, seconds=2):
    return Video(bvid="BV0000000001", title="字幕", cid=1, subtitle_status="available",
                 segments=[Segment(start=i * seconds, end=(i + 1) * seconds, text=text)
                           for i, text in enumerate(texts)])


def test_semantic_topic_change_is_a_chunk_boundary():
    video = video_with(["甲" * 25] * 20 + ["乙" * 25] * 20)
    chunks = split_subtitles(video, TopicEmbedding())
    assert any(chunk.end == 40 for chunk in chunks[:-1])
    assert all(200 <= len(chunk.text) <= 600 for chunk in chunks)
    assert all(c.bvid == video.bvid and c.cid == video.cid for c in chunks)


@pytest.mark.parametrize("size", [200, 600, 601, 799, 1201, 2201])
def test_single_line_and_long_line_always_respect_inclusive_limits(size):
    video = video_with(["甲" * size])
    chunks = split_subtitles(video, TopicEmbedding())
    assert all(200 <= len(c.text) <= 600 for c in chunks)
    assert all(c.start == 0 and c.end == 2 for c in chunks)


def test_short_tail_is_rebalanced_without_losing_original_lines():
    texts = [f"{i:03d}" + "甲" * 36 for i in range(17)]
    video = video_with(texts)
    chunks = split_subtitles(video, TopicEmbedding())
    assert len(chunks) == 2
    assert all(200 <= len(c.text) <= 600 for c in chunks)
    for line in video.segments:
        assert any(line.text in c.text and c.start <= line.start and c.end >= line.end
                   for c in chunks)
    assert split_subtitles(video, TopicEmbedding()) == chunks


def test_overlap_preserves_line_order_and_never_overflows():
    video = video_with([f"{i:03d}" + "甲" * 36 for i in range(40)])
    chunks = split_subtitles(video, TopicEmbedding())
    for chunk in chunks:
        ids = [int(line[:3]) for line in chunk.text.splitlines()]
        assert ids == list(range(ids[0], ids[-1] + 1))
        assert 200 <= len(chunk.text) <= 600
    assert {int(line[:3]) for c in chunks for line in c.text.splitlines()} == set(range(40))


def test_empty_and_too_short_subtitles_are_not_padded():
    assert split_subtitles(video_with([]), TopicEmbedding()) == []
    with pytest.raises(ValueError, match="200.*600"):
        split_subtitles(video_with(["甲" * 199]), TopicEmbedding())


def test_length_counts_newlines_and_mixed_text():
    video = video_with(["AI " * 33, "甲" * 100])
    chunks = split_subtitles(video, TopicEmbedding())
    assert len(chunks) == 1
    assert chunks[0].text == "AI " * 33 + "\n" + "甲" * 100
    assert len(chunks[0].text) == 200


def test_impossible_whole_line_partition_is_explicit():
    with pytest.raises(ValueError, match="200.*600"):
        split_subtitles(video_with(["甲" * 500, "乙" * 150]), TopicEmbedding())


def test_mixed_line_lengths_match_independent_feasibility_check():
    from functools import lru_cache
    from random import Random

    rng = Random(20261005)
    for _ in range(60):
        lengths = tuple(rng.choice([1, 50, 199, 200, 300, 400, 500, 600])
                        for _ in range(rng.randint(1, 8)))

        @lru_cache(None)
        def feasible(start):
            if start == len(lengths):
                return True
            return any(200 <= sum(lengths[start:end]) + end - start - 1 <= 600 and feasible(end)
                       for end in range(start + 1, len(lengths) + 1))

        video = video_with(["甲" * n for n in lengths])
        if feasible(0):
            chunks = split_subtitles(video, TopicEmbedding())
            assert all(200 <= len(c.text) <= 600 for c in chunks)
        else:
            with pytest.raises(ValueError, match="200.*600"):
                split_subtitles(video, TopicEmbedding())


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[0.0, 0.0]] * 4,
    [[1.0, np.nan]] * 4, [[1.0, np.inf]] * 4, [[1.0, 2.0], [1.0]] * 2])
def test_bad_boundary_vectors_fail_explicitly(vectors):
    class BadEmbedding:
        def embed(self, texts):
            return vectors

    with pytest.raises(ModelError):
        split_subtitles(video_with(["甲" * 25] * 40), BadEmbedding())
