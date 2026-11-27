"""Durable, immutable exact indexes with an atomic active-generation pointer."""

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .corpus import Source
from .dense import MAX_TOKENS, MODEL_ID, MODEL_REVISION
from .markdown import chunk_markdown


CHUNK_WORDS = 112
INDEX_CONTRACT = {
    "schema": 1, "chunk_words": CHUNK_WORDS, "model": MODEL_ID,
    "model_and_tokenizer_revision": MODEL_REVISION, "max_tokens": MAX_TOKENS,
    "encoding": "normalized-cls-float32", "dimension": 1024,
}
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation TEXT NOT NULL, input TEXT NOT NULL, state TEXT NOT NULL,
    token TEXT, expires REAL, previous TEXT,
    attempts INTEGER NOT NULL DEFAULT 0, stats TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS staging (token TEXT PRIMARY KEY, data TEXT NOT NULL, checksum TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, data TEXT NOT NULL, checksum TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def serialized(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def capture_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_bytes())
    repo, kb_id, releases = (manifest[key] for key in ("repo", "kb_id", "releases"))
    if not REPOSITORY.fullmatch(repo):
        raise ValueError("repo must be owner/name")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", kb_id):
        raise ValueError("Invalid kb_id")
    if not releases or manifest["default_release"] not in releases:
        raise ValueError("Default release must be present")
    if any(tag not in releases for tag in manifest["aliases"].values()):
        raise ValueError("Alias points to an unknown release")
    root = path.parent.resolve()
    documents = {}
    for tag, entries in sorted(releases.items()):
        if not re.fullmatch(r"v\d+(?:\.\d+)*", tag):
            raise ValueError(f"Unsupported release tag: {tag}")
        sources = {}
        for entry in entries:
            source_repo = entry.get("repo", repo)
            if not REPOSITORY.fullmatch(source_repo):
                raise ValueError("Source repo must be owner/name")
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts or not relative.name or str(relative) != entry["path"]:
                raise ValueError("Source path must be a canonical repository-relative path")
            if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
                raise ValueError("Source requires a full lowercase Git commit")
            local = (root / entry["snapshot"]).resolve()
            if not local.is_relative_to(root):
                raise ValueError("Document snapshot must stay under its manifest directory")
            raw = local.read_bytes()
            if digest(raw) != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {entry['path']}")
            name = str(relative)
            key = serialized([source_repo, name])
            if key in sources:
                raise ValueError(f"Duplicate document path: {source_repo}/{name}")
            sources[key] = {"repo": source_repo, "release": tag, "commit": entry["commit"], "path": name,
                            "text": raw.decode("utf-8"), "sha256": entry["sha256"]}
        documents[tag] = sources
    return {
        "scope": {"kb_id": kb_id, "default_release": manifest["default_release"],
                  "aliases": manifest["aliases"], "releases": documents},
        "contract": INDEX_CONTRACT,
    }


@dataclass(frozen=True)
class Attempt:
    job_id: int
    token: str
    lease_seconds: float


class SnapshotStore:
    def __init__(self, path: Path, clock=time.time) -> None:
        self.path = path
        self.clock = clock
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self, *, write: bool = False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _active(db) -> str | None:
        row = db.execute("SELECT value FROM state WHERE key='active'").fetchone()
        return row["value"] if row else None

    @staticmethod
    def _job(db, job_id: int):
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown job: {job_id}")
        return row

    def active_id(self) -> str | None:
        with self.connection() as db:
            return self._active(db)

    def enqueue(self, path: Path) -> int:
        corpus = capture_manifest(path)
        data = serialized(corpus)
        generation = digest(data)
        kb_id = corpus["scope"]["kb_id"]
        with self.connection(write=True) as db:
            db.execute("INSERT OR IGNORE INTO state VALUES ('kb_id', ?)", (kb_id,))
            if db.execute("SELECT value FROM state WHERE key='kb_id'").fetchone()[0] != kb_id:
                raise ValueError("This store belongs to another knowledge base")
            last = db.execute("SELECT id, generation, state FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
            if last and last["generation"] == generation and last["state"] != "superseded":
                return last["id"]
            return db.execute(
                "INSERT INTO jobs(generation,input,state) VALUES (?,?,'queued')", (generation, data),
            ).lastrowid

    def status(self, job_id: int) -> dict:
        with self.connection() as db:
            row = self._job(db, job_id)
            return {key: row[key] for key in ("id", "generation", "state", "attempts", "error")} | {
                "stats": json.loads(row["stats"]) if row["stats"] else None,
            }

    def claim(self, job_id: int, lease_seconds: float = 600) -> Attempt:
        if not 0 < lease_seconds < float("inf"):
            raise ValueError("Lease must be finite and positive")
        token = uuid4().hex
        with self.connection(write=True) as db:
            row = self._job(db, job_id)
            now = self.clock()
            if row["state"] in ("published", "superseded"):
                raise ValueError(f"Cannot claim {row['state']} job")
            if row["state"] == "running" and row["expires"] > now:
                raise ValueError("Job already holds an active lease")
            if row["token"]:
                db.execute("DELETE FROM staging WHERE token=?", (row["token"],))
            db.execute(
                """UPDATE jobs SET state='running', token=?, expires=?, previous=?,
                   attempts=attempts+1, stats=NULL, error=NULL WHERE id=?""",
                (token, now + lease_seconds, self._active(db), job_id),
            )
        return Attempt(job_id, token, lease_seconds)

    def renew(self, worker: Attempt) -> None:
        with self.connection(write=True) as db:
            self._require_lease(db, worker)
            db.execute("UPDATE jobs SET expires=? WHERE id=?", (self.clock() + worker.lease_seconds, worker.job_id))

    def _require_lease(self, db, worker: Attempt):
        row = self._job(db, worker.job_id)
        if row["state"] != "running" or row["token"] != worker.token or row["expires"] <= self.clock():
            raise ValueError("Worker no longer owns the lease")
        return row

    @staticmethod
    def _read_checked(row):
        if digest(row["data"]) != row["checksum"]:
            raise ValueError("Index checksum mismatch")
        return json.loads(row["data"])

    def stage(self, worker: Attempt, encode) -> dict:
        import numpy as np

        with self.connection() as db:
            job = self._require_lease(db, worker)
            corpus = json.loads(job["input"])
            if corpus["contract"] != INDEX_CONTRACT:
                raise ValueError("Job was prepared for a different index contract")
            previous = db.execute("SELECT * FROM snapshots WHERE id=?", (job["previous"],)).fetchone()
            old = self._read_checked(previous)["documents"] if previous else {}
        contract = serialized(INDEX_CONTRACT)
        documents, pending = {}, {}
        for tag, release in corpus["scope"]["releases"].items():
            for record in release.values():
                doc_id = serialized([tag, record["repo"], record["path"]])
                source = Source(**{key: val for key, val in record.items() if key != "sha256"})
                indexed = []
                for chunk in chunk_markdown(source.text, CHUNK_WORDS):
                    key = digest(serialized([contract, chunk.retrieval_text]))
                    pending[key] = chunk.retrieval_text
                    indexed.append({"chunk": asdict(chunk), "vector": key})
                documents[doc_id] = {"source": record, "chunks": indexed}

        with self.connection() as db:
            cached = {key for key in pending if db.execute("SELECT 1 FROM vectors WHERE key=?", (key,)).fetchone()}
        missing = [key for key in pending if key not in cached]
        for start in range(0, len(missing), 16):
            keys = missing[start:start + 16]
            self.renew(worker)
            vectors = np.asarray(encode([pending[key] for key in keys]), dtype="<f4")
            if vectors.shape != (len(keys), INDEX_CONTRACT["dimension"]) or not np.all(np.isfinite(vectors)):
                raise ValueError("Encoder output has invalid shape or values")
            with self.connection(write=True) as db:
                self._require_lease(db, worker)
                db.executemany(
                    "INSERT OR IGNORE INTO vectors VALUES (?,?)",
                    ((key, vector.tobytes()) for key, vector in zip(keys, vectors, strict=True)),
                )
        scope = {**corpus["scope"], "releases": list(corpus["scope"]["releases"])}
        data = serialized({"scope": scope, "contract": INDEX_CONTRACT, "documents": documents})
        stats = {
            "documents_added": len(documents.keys() - old.keys()),
            "documents_deleted": len(old.keys() - documents.keys()),
            "documents_changed": sum(documents[key] != old[key] for key in documents.keys() & old.keys()),
            "encoded": len(missing), "reused": len(cached),
            "chunks_written": sum(len(doc["chunks"]) for doc in documents.values()),
        }
        with self.connection(write=True) as db:
            self._require_lease(db, worker)
            db.execute("INSERT OR REPLACE INTO staging VALUES (?,?,?)", (worker.token, data, digest(data)))
            db.execute("UPDATE jobs SET stats=? WHERE id=?", (serialized(stats), worker.job_id))
        return stats

    def publish(self, worker: Attempt) -> bool:
        with self.connection(write=True) as db:
            row = self._job(db, worker.job_id)
            if row["state"] != "running" or row["token"] != worker.token or row["expires"] <= self.clock():
                return False
            latest = db.execute("SELECT MAX(id) FROM jobs").fetchone()[0]
            if latest != worker.job_id or row["previous"] != self._active(db):
                db.execute("UPDATE jobs SET state='superseded' WHERE id=?", (worker.job_id,))
                db.execute("DELETE FROM staging WHERE token=?", (worker.token,))
                return False
            stage = db.execute("SELECT * FROM staging WHERE token=?", (worker.token,)).fetchone()
            if stage is None:
                raise ValueError("Job has no staged index")
            data = self._read_checked(stage)
            required = {hit["vector"] for doc in data["documents"].values() for hit in doc["chunks"]}
            for key in required:
                vector = db.execute("SELECT length(data) FROM vectors WHERE key=?", (key,)).fetchone()
                if not vector or vector[0] != 4 * INDEX_CONTRACT["dimension"]:
                    raise ValueError("Staged snapshot refers to an invalid vector")
            db.execute("INSERT OR IGNORE INTO snapshots VALUES (?,?,?)",
                       (row["generation"], stage["data"], stage["checksum"]))
            db.execute("INSERT OR REPLACE INTO state VALUES ('active',?)", (row["generation"],))
            db.execute("UPDATE jobs SET state='published' WHERE id=?", (worker.job_id,))
            db.execute("DELETE FROM staging WHERE token=?", (worker.token,))
            return True

    def run(self, job_id: int, encode) -> dict:
        status = self.status(job_id)
        if status["state"] in ("published", "superseded"):
            return status
        worker = self.claim(job_id)
        try:
            self.stage(worker, encode)
            if not self.publish(worker) and self.status(job_id)["state"] != "superseded":
                raise ValueError("Snapshot lease expired before publication")
        except Exception as exc:
            with self.connection(write=True) as db:
                db.execute(
                    "UPDATE jobs SET state='failed', error=? WHERE id=? AND token=? AND state='running'",
                    (str(exc), job_id, worker.token),
                )
                db.execute("DELETE FROM staging WHERE token=?", (worker.token,))
            raise
        return self.status(job_id)

    def read(self, generation_id: str):
        import numpy as np

        with self.connection() as db:
            row = db.execute("SELECT * FROM snapshots WHERE id=?", (generation_id,)).fetchone()
            if row is None:
                raise KeyError(generation_id)
            snapshot = self._read_checked(row)
            keys = {hit["vector"] for doc in snapshot["documents"].values() for hit in doc["chunks"]}
            vectors = {}
            for key in keys:
                blob = db.execute("SELECT data FROM vectors WHERE key=?", (key,)).fetchone()
                if blob is None or len(blob[0]) != 4 * INDEX_CONTRACT["dimension"]:
                    raise ValueError("Published vector is unavailable or has an invalid length")
                vectors[key] = np.frombuffer(blob[0], dtype="<f4")
        return snapshot, vectors
