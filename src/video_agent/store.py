import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

from .schemas import Chunk, Profile, Video, utc_now


class RunBusy(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS videos (bvid TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY, bvid TEXT NOT NULL, text TEXT NOT NULL,
                extracted TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS clicks (
                id INTEGER PRIMARY KEY, bvid TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS indexes (
                bvid TEXT PRIMARY KEY, content_hash TEXT NOT NULL, space_id TEXT NOT NULL,
                summary TEXT NOT NULL, evidence_ids TEXT NOT NULL, vector TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, bvid TEXT NOT NULL, payload TEXT NOT NULL,
                space_id TEXT NOT NULL, vector TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                date TEXT PRIMARY KEY, status TEXT NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT, payload TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS recommendations (
                date TEXT NOT NULL, bvid TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(date, bvid));
        """)

    def close(self):
        with self.lock:
            self.connection.close()

    def profile(self) -> Profile | None:
        with self.lock:
            row = self.connection.execute("SELECT payload FROM profile WHERE id=1").fetchone()
        return Profile.model_validate_json(row[0]) if row else None

    def save_profile(self, profile: Profile):
        with self.lock, self.connection:
            self.connection.execute("INSERT OR REPLACE INTO profile VALUES(1, ?)",
                                    (profile.model_dump_json(),))

    def save_video(self, video: Video):
        with self.lock, self.connection:
            self.connection.execute("INSERT OR REPLACE INTO videos VALUES(?, ?)",
                                    (video.bvid, video.model_dump_json()))

    def video(self, bvid: str) -> Video | None:
        with self.lock:
            row = self.connection.execute("SELECT payload FROM videos WHERE bvid=?",
                                          (bvid,)).fetchone()
        return Video.model_validate_json(row[0]) if row else None

    def videos(self) -> list[Video]:
        with self.lock:
            rows = self.connection.execute("SELECT payload FROM videos ORDER BY bvid").fetchall()
        return [Video.model_validate_json(row[0]) for row in rows]

    def add_feedback(self, bvid: str, text: str, at: datetime | None = None) -> int:
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO feedback(bvid,text,created_at) VALUES(?,?,?)",
                (bvid, text, (at or utc_now()).isoformat()))
            return cursor.lastrowid

    def set_extraction(self, feedback_id: int, extracted: dict):
        with self.lock, self.connection:
            self.connection.execute("UPDATE feedback SET extracted=? WHERE id=?",
                                    (json.dumps(extracted, ensure_ascii=False), feedback_id))

    def feedback(self, since: datetime | None = None) -> list[dict]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM feedback WHERE created_at>=? ORDER BY created_at,id",
                ((since.isoformat() if since else ""),)).fetchall()
        return [{**dict(row), "extracted": json.loads(row["extracted"]) if row["extracted"]
                 else None} for row in rows]

    def add_click(self, bvid: str, at: datetime | None = None):
        if not self.video(bvid):
            raise ValueError("Unknown video")
        with self.lock, self.connection:
            self.connection.execute("INSERT INTO clicks(bvid,created_at) VALUES(?,?)",
                                    (bvid, (at or utc_now()).isoformat()))

    def clicks(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT bvid, created_at FROM clicks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def excluded(self, at: datetime) -> set[str]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT bvid FROM recommendations WHERE created_at>? AND created_at<=?",
                ((at - timedelta(hours=168)).isoformat(), at.isoformat())).fetchall()
        result = {row[0] for row in rows}
        for feedback in self.feedback():
            data = feedback["extracted"] or {}
            if data.get("ban") or data.get("watched"):
                result.add(feedback["bvid"])
        return result

    def index(self, bvid: str) -> dict | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM indexes WHERE bvid=?", (bvid,)).fetchone()
        if not row:
            return None
        return {**dict(row), "vector": json.loads(row["vector"]),
                "evidence_ids": json.loads(row["evidence_ids"])}

    def index_spaces(self) -> set[str]:
        with self.lock:
            rows = self.connection.execute("SELECT DISTINCT space_id FROM indexes").fetchall()
        return {row[0] for row in rows}

    def save_index(self, video: Video, content_hash: str, space_id: str, summary: str,
                   evidence_ids: list[str], vector: list[float],
                   chunks: list[Chunk], vectors: list[list[float]]):
        if len(chunks) != len(vectors):
            raise ValueError("Chunk/vector count differs")
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM chunks WHERE bvid=?", (video.bvid,))
            self.connection.execute("INSERT OR REPLACE INTO indexes VALUES(?,?,?,?,?,?)", (
                video.bvid, content_hash, space_id, summary, json.dumps(evidence_ids),
                json.dumps(vector),
            ))
            self.connection.executemany("INSERT INTO chunks VALUES(?,?,?,?,?)", [
                (chunk.id, video.bvid, chunk.model_dump_json(), space_id, json.dumps(embedding))
                for chunk, embedding in zip(chunks, vectors)
            ])

    def indexed_chunks(self, bvid: str) -> list[tuple[Chunk, list[float], str]]:
        with self.lock:
            rows = self.connection.execute("SELECT * FROM chunks WHERE bvid=?", (bvid,)).fetchall()
        return [(Chunk.model_validate_json(row["payload"]), json.loads(row["vector"]),
                 row["space_id"]) for row in rows]

    def remove_index(self, bvid: str):
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM chunks WHERE bvid=?", (bvid,))
            self.connection.execute("DELETE FROM indexes WHERE bvid=?", (bvid,))

    def claim_run(self, date: str, at: datetime) -> dict | None:
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute("SELECT * FROM runs WHERE date=?", (date,)).fetchone()
                if row and row["status"] == "success":
                    payload = json.loads(row["payload"])
                    if payload["items"]:
                        self.connection.commit()
                        return payload
                if row and row["status"] == "running" and (
                    at - datetime.fromisoformat(row["started_at"]) < timedelta(hours=1)
                ):
                    raise RunBusy("A recommendation run for this date is already running")
                self.connection.execute(
                    "INSERT OR REPLACE INTO runs VALUES(?, 'running', ?, NULL, NULL, NULL)",
                    (date, at.isoformat()))
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
        return None

    def complete_run(self, date: str, payload: dict, at: datetime):
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE runs SET status='success',finished_at=?,payload=?,error=NULL "
                "WHERE date=? AND status='running' AND started_at=?",
                (utc_now().isoformat(), json.dumps(payload, ensure_ascii=False), date, at.isoformat()))
            if cursor.rowcount != 1:
                raise RunBusy("Recommendation run ownership changed; result was not published")
            self.connection.executemany("INSERT INTO recommendations VALUES(?,?,?)",
                                        [(date, item["bvid"], at.isoformat())
                                         for item in payload["items"]])

    def fail_run(self, date: str, error: str, at: datetime):
        with self.lock, self.connection:
            self.connection.execute("UPDATE runs SET status='failed',finished_at=?,error=? "
                                    "WHERE date=? AND status='running' AND started_at=?",
                                    (utc_now().isoformat(), error, date, at.isoformat()))

    def run(self, date: str) -> dict | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM runs WHERE date=?", (date,)).fetchone()
        if not row:
            return None
        return {**dict(row), "payload": json.loads(row["payload"]) if row["payload"] else None}
