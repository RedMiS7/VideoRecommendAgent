from datetime import timedelta

import pytest

from video_agent.memory import Memory
from video_agent.models.contracts import ModelError
from video_agent.schemas import FeedbackInput, Preference, Profile, Video, utc_now
from video_agent.store import Store


class FeedbackModel:
    def __init__(self, direction="avoid", scope="global", ambiguous=False):
        self.direction, self.scope, self.ambiguous = direction, scope, ambiguous

    def generate_json(self, messages, schema):
        return {"observations": [{"dimension": "depth", "value": "beginner",
                "direction": self.direction, "scope": self.scope,
                "scope_value": "video" if self.scope == "video" else None}],
                "ambiguous": self.ambiguous}


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "agent.sqlite")
    result.save_profile(Profile(topics=["RAG"]))
    for i in range(1, 5):
        result.save_video(Video(bvid=f"BV{i:010d}", title="RAG"))
    yield result
    result.close()


def submit(memory, number, at=None):
    return memory.submit(FeedbackInput(bvid=f"BV{number:010d}", text="以后不要太基础的"), at)


def test_three_distinct_videos_promote_and_survive_restart(store):
    memory = Memory(store, FeedbackModel())
    submit(memory, 1)
    submit(memory, 1)
    submit(memory, 2)
    assert not store.profile().preferences
    submit(memory, 3)
    assert store.profile().preferences[0].source == "inferred"
    path = store.connection.execute("PRAGMA database_list").fetchone()[2]
    reopened = Store(path)
    assert reopened.profile().preferences[0].value == "beginner"
    reopened.close()


def test_video_specific_and_ambiguous_feedback_do_not_promote(store):
    for model in [FeedbackModel(scope="video"), FeedbackModel(ambiguous=True)]:
        memory = Memory(store, model)
        for i in range(1, 4):
            submit(memory, i)
    assert not store.profile().preferences


def test_conflict_and_old_feedback_do_not_promote(store):
    memory = Memory(store, FeedbackModel())
    submit(memory, 1, utc_now() - timedelta(days=31))
    submit(memory, 2)
    Memory(store, FeedbackModel(direction="prefer")).submit(
        FeedbackInput(bvid="BV0000000003", text="我喜欢基础"))
    submit(memory, 4)
    assert not store.profile().preferences


def test_explicit_preference_is_not_overridden(store):
    profile = store.profile()
    profile.preferences = [Preference(dimension="depth", value="beginner", direction="prefer",
                                      scope="global", source="explicit")]
    store.save_profile(profile)
    memory = Memory(store, FeedbackModel())
    for i in range(1, 4):
        submit(memory, i)
    assert store.profile().preferences[0].direction == "prefer"


@pytest.mark.parametrize("scope_value", [None, "", "BV0000000004", "missing"])
def test_video_observation_is_bound_to_feedback_video_before_validation(store, scope_value):
    class Model(FeedbackModel):
        def generate_json(self, messages, schema):
            result = super().generate_json(messages, schema)
            observation = result["observations"][0]
            if scope_value == "missing":
                observation.pop("scope_value")
            else:
                observation["scope_value"] = scope_value
            return result

    result = submit(Memory(store, Model(scope="video")), 1)
    observation = result["extracted"]["observations"][0]
    assert observation["scope"] == "video"
    assert observation["scope_value"] == "BV0000000001"
    assert store.feedback()[0]["extracted"]["observations"] == [observation]
    assert not store.profile().preferences


@pytest.mark.parametrize("scope,scope_value", [("topic", None), ("global", "RAG")])
def test_invalid_scope_is_reported_as_model_error_and_raw_feedback_is_preserved(
    store, scope, scope_value
):
    class Model(FeedbackModel):
        def generate_json(self, messages, schema):
            result = super().generate_json(messages, schema)
            result["observations"][0]["scope_value"] = scope_value
            return result

    with pytest.raises(ModelError) as error:
        submit(Memory(store, Model(scope=scope)), 1)
    assert error.value.kind == "invalid_output"
    assert "scope_value" not in str(error.value)
    feedback = store.feedback()[0]
    assert feedback["text"] == "以后不要太基础的" and feedback["extracted"] is None
    assert not store.profile().preferences


def test_topic_observation_keeps_model_supplied_topic(store):
    class Model(FeedbackModel):
        def generate_json(self, messages, schema):
            result = super().generate_json(messages, schema)
            result["observations"][0]["scope_value"] = "RAG"
            return result

    result = submit(Memory(store, Model(scope="topic")), 1)
    assert result["extracted"]["observations"][0]["scope_value"] == "RAG"
