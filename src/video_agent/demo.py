"""Explicit offline test doubles. These are not real model or Bilibili integrations."""
import hashlib
import json
import re

import numpy as np

from .schemas import Segment, Video, VideoStats


class DemoEmbedding:
    space_id = "offline-test:char-bigrams:128:v1"

    def embed(self, texts):
        result = []
        for text in texts:
            text = re.sub(r"\s+", "", text.lower())
            vector = np.zeros(128)
            for i in range(max(1, len(text) - 1)):
                token = text[i:i + 2]
                slot = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 128
                vector[slot] += 1
            result.append(vector.tolist())
        return result


class DemoLanguageModel:
    def generate_json(self, messages, schema):
        task = messages[0].content.splitlines()[0].removeprefix("task:")
        if task == "init":
            return {"topics": ["RAG"], "goal": messages[-1].content, "level": "入门"}
        data = json.loads(messages[-1].content)
        if task == "queries":
            return {"queries": data["context"]["profile"]["topics"][:3]}
        if task == "summary":
            content = data["content"]
            ids = [c["id"] for c in content if "id" in c]
            if not ids:
                ids = [identifier for c in content for identifier in c["evidence_ids"]]
            return {"summary": " ".join(c.get("text", c.get("summary", "")) for c in content)[:400],
                    "evidence_ids": ids[:3]}
        if task == "feedback":
            text = data["feedback"]
            scope = "global" if any(word in text for word in ("以后", "所有", "整体")) else "video"
            scope_value = None if scope == "global" else data["bvid"]
            observations = []
            for keyword, dimension, value, direction in [
                ("太基础", "depth", "beginner", "avoid"),
                ("进阶", "depth", "advanced", "prefer"),
                ("太长", "duration", "long", "avoid"),
                ("太慢", "pace", "slow", "avoid"),
                ("太快", "pace", "fast", "avoid"),
                ("喜欢实操", "style", "practical", "prefer"),
                ("喜欢理论", "style", "theoretical", "prefer"),
                ("喜欢基础", "depth", "beginner", "prefer"),
                ("不喜欢理论", "style", "theoretical", "avoid"),
            ]:
                if keyword in text and not (keyword == "喜欢理论" and "不喜欢理论" in text):
                    observations.append({"dimension": dimension, "value": value,
                        "direction": direction, "scope": scope, "scope_value": scope_value})
            return {"observations": observations, "watched": "看完了" in text,
                    "ban": "不再推荐这个" in text, "ambiguous": not observations}
        if task in ("screen", "rank"):
            context = data["context"]
            profile = context.get("profile") or {}
            items = []
            for candidate in data["candidates"]:
                chunks = candidate.get("chunks", [])
                text = (" ".join(c["text"] for c in chunks) if chunks else " ".join(
                    str(candidate.get(k, "")) for k in ("title", "description", "search_summary")))
                topics = profile.get("topics", [])
                matched = not topics or any(t.lower() in text.lower() for t in topics)
                score = 0.7 if matched else 0.1
                preferences = list(profile.get("preferences", []))
                for feedback in context.get("video_feedback", []):
                    if feedback["bvid"] == candidate["bvid"]:
                        preferences.extend((feedback.get("extracted") or {}).get("observations", []))
                markers = {"beginner": ("基础", "入门"), "advanced": ("进阶", "深入"),
                           "practical": ("实操", "案例", "代码"), "theoretical": ("理论", "推导")}
                for preference in preferences:
                    if preference.get("scope") == "video" and (
                        preference.get("scope_value") != candidate["bvid"]
                    ):
                        continue
                    if preference.get("scope") == "topic" and (
                        str(preference.get("scope_value", "")) not in text
                    ):
                        continue
                    values = markers.get(preference["value"], (preference["value"],))
                    if any(value in text for value in values):
                        if preference["direction"] == "avoid":
                            matched, score = False, 0.1
                        else:
                            score = min(1., score + 0.2)
                reason = ("字幕片段：" + chunks[0]["text"][:200]) if chunks else (
                    "标题和摘要与学习主题匹配：" + candidate["title"])
                items.append({"bvid": candidate["bvid"], "score": score, "matched": matched,
                              "reason": reason, "evidence_ids": [chunks[0]["id"]] if chunks else []})
            return {"items": items}
        if task == "judge":
            return {"supported": any(c["text"][:200] in data["reason"]
                                     for c in data.get("chunks", [])),
                    "note": "Deterministic test-double check, not an AI quality assessment"}
        raise ValueError(f"Unsupported offline task: {task}")


def demo_videos(count=45):
    result = []
    for i in range(1, count + 1):
        level = "基础入门" if i % 2 else "进阶深入"
        result.append(Video(bvid=f"BV{i:010d}", title=f"合成测试 RAG {level} {i}", cid=i,
            description=f"RAG {level} 检索增强生成工程案例", duration=600,
            stats=VideoStats(views=1000 + i, likes=50, favorites=60, comments=10),
            subtitle_status="absent" if i % 7 == 0 else "available",
            segments=[] if i % 7 == 0 else [
                Segment(start=0, end=60, text=f"RAG {level}，介绍检索增强生成原理。"
                    "先明确用户的问题，再从知识库寻找相关资料，把资料与问题一起交给语言模型。"
                    "文档需要保留来源信息，向量用于表示文本的语义，检索结果作为回答的上下文。"
                    "这个过程让回答有可以检查的依据，用户能够回到原始资料查看相关内容。"),
                Segment(start=60, end=120, text="代码实操案例：实现向量检索、缓存和结果评估。"
                    "先按连续内容生成文档片段，再调用向量模型计算表示，并保存到本地索引。"
                    "检索时计算问题向量，比较相似度，选择相关片段供模型生成带引用的回答。"
                    "记录每次请求的耗时，检查引用是否存在，并对已有索引进行缓存以减少重复计算。")]))
    return result


class DemoSource:
    def __init__(self, videos=None):
        self.videos = videos if videos is not None else demo_videos()

    def search(self, query):
        return self.videos

    def metadata(self, video):
        return video

    def subtitles(self, video):
        return video

    def close(self):
        pass
