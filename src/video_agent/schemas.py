from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(Record):
    dimension: Literal["topic", "depth", "style", "duration", "pace"]
    value: str = Field(min_length=1, max_length=200)
    direction: Literal["prefer", "avoid"]
    scope: Literal["video", "topic", "global"]
    scope_value: str | None = None

    @model_validator(mode="after")
    def require_scope_value(self):
        if self.scope != "global" and not self.scope_value:
            raise ValueError("Video/topic observations require scope_value")
        if self.scope == "global" and self.scope_value is not None:
            raise ValueError("Global observations cannot have scope_value")
        return self


class Preference(Observation):
    source: Literal["explicit", "inferred"] = "explicit"
    evidence_ids: list[int] = Field(default_factory=list)


class Profile(Record):
    topics: list[str] = Field(min_length=1, max_length=10)
    goal: str = Field(default="", max_length=2000)
    level: str = Field(default="未指定", max_length=100)
    max_duration_minutes: int | None = Field(default=None, ge=1, le=600)
    preferences: list[Preference] = Field(default_factory=list)
    daily_count: int = Field(default=5, ge=1, le=5)
    daily_time: str = Field(default="08:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")

    @model_validator(mode="after")
    def nonempty_topics(self):
        if any(not topic.strip() for topic in self.topics):
            raise ValueError("Topics cannot be empty")
        return self


class FeedbackExtraction(Record):
    observations: list[Observation] = Field(default_factory=list)
    watched: bool = False
    ban: bool = False
    ambiguous: bool = False


class FeedbackInput(Record):
    bvid: str = Field(pattern=r"^BV[0-9A-Za-z]{10}$")
    text: str = Field(min_length=1, max_length=4000)


class Segment(Record):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def time_order(self):
        if self.end < self.start:
            raise ValueError("Subtitle end precedes start")
        return self


class VideoStats(Record):
    views: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    favorites: int | None = Field(default=None, ge=0)
    comments: int | None = Field(default=None, ge=0)


class Video(Record):
    bvid: str = Field(pattern=r"^BV[0-9A-Za-z]{10}$")
    title: str
    description: str = ""
    search_summary: str = ""
    uploader: str = ""
    duration: int | None = Field(default=None, ge=0)
    published_at: int | None = None
    cid: int | None = None
    part_count: int = 1
    part_title: str = ""
    stats: VideoStats = Field(default_factory=VideoStats)
    metadata_at: datetime | None = None
    subtitle_status: Literal["unknown", "available", "absent", "error"] = "unknown"
    subtitle_error: str | None = None
    subtitle_checked_at: datetime | None = None
    segments: list[Segment] = Field(default_factory=list)

    @property
    def url(self):
        return f"https://www.bilibili.com/video/{self.bvid}/?p=1"


class Chunk(Record):
    id: str
    bvid: str
    cid: int
    start: float
    end: float
    text: str


class Summary(Record):
    summary: str
    evidence_ids: list[str]


class Score(Record):
    bvid: str
    score: float = Field(ge=0, le=1)
    matched: bool
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)


class Scores(Record):
    items: list[Score]


class Queries(Record):
    queries: list[str] = Field(min_length=1, max_length=3)
