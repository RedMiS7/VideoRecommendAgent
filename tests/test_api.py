from fastapi.testclient import TestClient

from video_agent.api import create_app
from video_agent.config import Settings
from video_agent.runtime import Runtime


def test_profile_draft_and_recommendation_feedback_flow(tmp_path):
    runtime = Runtime(Settings(data_dir=tmp_path), offline=True)
    with TestClient(create_app(runtime)) as client:
        assert client.get("/profile").status_code == 404
        draft = client.post("/profile/draft", json={"text": "学习RAG工程实践"})
        assert draft.status_code == 200
        assert client.get("/profile").status_code == 404  # Draft does not activate memory.
        assert client.put("/profile", json=draft.json()).status_code == 200
        first = client.post("/recommendations/run")
        assert first.status_code == 200
        assert len(first.json()["items"]) == 5
        item = first.json()["items"][0]
        assert item["click_url"].startswith("http://testserver/click/")
        assert client.get("/recommendations").json()["payload"] == first.json()
        feedback = client.post("/feedback", json={"bvid": item["bvid"], "text": "这个太基础"})
        assert feedback.status_code == 200
        assert client.get("/memory").json()["feedback"][0]["text"] == "这个太基础"
        assert client.get(item["click_url"]).status_code == 409  # Synthetic URLs cannot open.
        assert client.get("/click/unknown").status_code == 404
        assert client.get("/recommendations?date=invalid").status_code == 422
    runtime.close()


def test_click_redirects_only_to_stored_video(tmp_path):
    from video_agent.demo import demo_videos

    runtime = Runtime(Settings(data_dir=tmp_path), offline=True)
    runtime.offline = False
    video = demo_videos(1)[0]
    runtime.store.save_video(video)
    with TestClient(create_app(runtime)) as client:
        response = client.get(f"/click/{video.bvid}", follow_redirects=False)
        assert response.status_code == 302
        assert response.headers["location"] == video.url
        assert len(runtime.store.clicks()) == 1
        assert client.get("/click/https://evil.test").status_code == 404
    runtime.offline = True
    runtime.close()
