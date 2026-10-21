import httpx
import pytest

from video_agent.models.contracts import Message, ModelError
from video_agent.models.qwen import QwenEmbeddingModel, QwenLanguageModel


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_structured_output_is_validated_and_repaired():
    responses = iter(['{"value": "wrong"}', '{"value": 3}'])
    model = QwenLanguageModel("test", "https://model.test/v1", "model", client(
        lambda request: httpx.Response(200, json={
            "choices": [{"message": {"content": next(responses)}}]
        })
    ))
    result = model.generate_json([Message("user", "test")], {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    })
    assert result == {"value": 3}


@pytest.mark.parametrize("status,kind", [(401, "authentication"), (429, "rate_limit"),
                                         (503, "unavailable")])
def test_provider_errors_are_normalized(status, kind):
    model = QwenLanguageModel("test", "https://model.test/v1", "model",
                              client(lambda r: httpx.Response(status)))
    with pytest.raises(ModelError) as error:
        model.generate_json([Message("user", "test")], {"type": "object"})
    assert error.value.kind == kind
    assert "test" not in str(error.value)


def test_embeddings_restore_input_order():
    model = QwenEmbeddingModel("test", "https://model.test/v1", "embed", client(
        lambda r: httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0., 1.] + [0.] * 1022},
            {"index": 0, "embedding": [1.] + [0.] * 1023}
        ]})
    ))
    assert model.embed(["a", "b"]) == [[1.] + [0.] * 1023, [0., 1.] + [0.] * 1022]


def test_malformed_embedding_count_is_rejected():
    model = QwenEmbeddingModel("test", "https://model.test/v1", "embed",
                              client(lambda r: httpx.Response(200, json={"data": []})))
    with pytest.raises(ModelError, match="embedding"):
        model.embed(["a"])


def test_wrong_embedding_dimension_is_rejected():
    model = QwenEmbeddingModel("test", "https://model.test/v1", "embed", client(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1., 0.]}]})
    ))
    with pytest.raises(ModelError, match="embedding"):
        model.embed(["a"])
