"""Structured JSON logging of every Q&A turn to CloudWatch Logs.

One JSON object per log event, so you can query it in CloudWatch Logs Insights, e.g.:

    fields @timestamp, question, total_ms, chunk_count, input_tokens, output_tokens
    | filter event = "qa"
    | sort @timestamp desc

    # slowest turns
    | stats avg(llm_ms), max(llm_ms) by bin(1h)

    # retrieval quality - the chunks array is flattened to chunks.0.*, chunks.1.*, ...
    # (unnest does not work on auto-discovered JSON arrays; it yields empty grouping keys).
    # score is cosine similarity, so higher is better - always pin the schema, because
    # qa/1 events logged score as squared L2 distance instead.
    | filter event = "qa" and schema = "qa/2"
    | stats count(*), avg(chunks.0.score) by chunks.0.chunk_id, chunks.0.page

Configuration (all optional - env vars, or Streamlit secrets copied into env by the frontend):
    PRV_LOG_GROUP         CloudWatch log group. Default "/privacy-act-rag/qa". Set to "" to disable.
    PRV_LOG_RETENTION_DAYS  Retention applied when the group is created. Default 30. "0" = never expire.
    PRV_LOG_CHUNK_CHARS   Chars of each chunk's text to log. Default 800. "0" = omit text, "full" = all.
    PRV_LOG_ANSWER_CHARS  Chars of the answer to log. Default "full".
    PRV_QA_LOG_FILE       Also append each event to this local JSONL file (handy offline / for tests).

Logging is best-effort and asynchronous: it runs on a background thread and every failure is
swallowed (reported once to stderr) so a missing IAM permission can never break the app.

Required IAM permissions on the log group:
    logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents, logs:PutRetentionPolicy
"""

import atexit
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import boto3

LOG_GROUP = os.environ.get("PRV_LOG_GROUP", "/privacy-act-rag/qa")
RETENTION_DAYS = int(os.environ.get("PRV_LOG_RETENTION_DAYS", "30"))
LOCAL_LOG_FILE = os.environ.get("PRV_QA_LOG_FILE", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# CloudWatch rejects events over 256 KB (including a 26-byte overhead); stay well clear.
MAX_EVENT_BYTES = 200_000


def _char_limit(env_name, default):
    raw = os.environ.get(env_name, default).strip().lower()
    return None if raw == "full" else int(raw)


CHUNK_CHARS = _char_limit("PRV_LOG_CHUNK_CHARS", "800")
ANSWER_CHARS = _char_limit("PRV_LOG_ANSWER_CHARS", "full")


def _truncate(text, limit):
    # Returns (text, was_truncated). limit None = keep everything, 0 = drop the text entirely.
    if limit is None:
        return text, False
    if limit <= 0:
        return None, True
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def chunk_id(text):
    """Stable short id for a chunk, so repeated retrievals of the same text group together."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def cosine_from_l2_sq(l2_sq):
    """Convert FAISS's squared L2 distance to cosine similarity.

    IndexFlatL2 returns *squared* L2 distance. Titan v2 normalises both document and query
    embeddings to unit length, and for unit vectors ||q-d||^2 == 2 - 2*cos(q,d), so the two
    metrics are an exact monotonic bijection: rebuilding the index with an inner-product
    metric would retrieve the same chunks in the same order (verified against all 860 vectors -
    Spearman rank correlation exactly 1.0 over 5 queries). We log cosine instead of the raw
    distance only because it reads better: higher = more similar, bounded to [0, 1] for text,
    and comparable across questions.

    This conversion is only valid while the embeddings are unit-norm. If EMBEDDING_MODEL_ID
    ever changes to a model that does not normalise, rebuild the index with
    DistanceStrategy.MAX_INNER_PRODUCT + normalize_L2=True and log the score directly.
    """
    return max(-1.0, min(1.0, 1.0 - float(l2_sq) / 2.0))


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _CloudWatchSink:
    """Lazily creates the log group/stream, then fires put_log_events off the main thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._client = None
        self._stream = None
        self._ready = False
        self._broken = False
        # maxsize-free single worker keeps events in order and bounds concurrency to 1 request.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qa-log")
        atexit.register(self.shutdown)

    def _warn_once(self, message):
        if not self._broken:
            self._broken = True
            print(f"[rag_logging] disabled: {message}", file=sys.stderr)

    def _ensure_stream(self):
        if self._ready:
            return True
        if self._broken:
            return False
        with self._lock:
            if self._ready:
                return True
            try:
                self._client = boto3.Session(region_name=AWS_REGION).client("logs")
                try:
                    self._client.create_log_group(logGroupName=LOG_GROUP)
                    if RETENTION_DAYS > 0:
                        self._client.put_retention_policy(
                            logGroupName=LOG_GROUP, retentionInDays=RETENTION_DAYS)
                except self._client.exceptions.ResourceAlreadyExistsException:
                    pass
                # One stream per process: date prefix sorts nicely, uuid keeps replicas apart.
                self._stream = f"{datetime.now(timezone.utc):%Y/%m/%d}/{uuid.uuid4().hex[:12]}"
                self._client.create_log_stream(logGroupName=LOG_GROUP, logStreamName=self._stream)
                self._ready = True
                return True
            except Exception as e:
                self._warn_once(f"could not open log stream {LOG_GROUP} ({e})")
                return False

    def _put(self, message):
        if not self._ensure_stream():
            return
        try:
            # PutLogEvents no longer needs a sequence token (deprecated by CloudWatch in 2023).
            self._client.put_log_events(
                logGroupName=LOG_GROUP,
                logStreamName=self._stream,
                logEvents=[{"timestamp": int(time.time() * 1000), "message": message}],
            )
        except Exception as e:
            self._warn_once(f"put_log_events failed ({e})")

    def submit(self, message):
        try:
            self._executor.submit(self._put, message)
        except RuntimeError:
            pass  # interpreter shutting down

    def shutdown(self):
        self._executor.shutdown(wait=True)


_sink = _CloudWatchSink() if LOG_GROUP else None


def _serialize(record):
    """JSON-encode the record, shrinking chunk texts if the event would exceed the CW size cap."""
    message = json.dumps(record, ensure_ascii=False, default=str)
    if len(message.encode("utf-8")) <= MAX_EVENT_BYTES or not record.get("chunks"):
        return message
    for limit in (400, 120, 0):
        for chunk in record["chunks"]:
            if chunk.get("text") is not None:
                chunk["text"], chunk["text_truncated"] = _truncate(chunk["text"], limit)
        record["size_capped"] = True
        message = json.dumps(record, ensure_ascii=False, default=str)
        if len(message.encode("utf-8")) <= MAX_EVENT_BYTES:
            break
    return message


def emit(record):
    """Send one JSON record to CloudWatch (and the local JSONL mirror, if configured)."""
    record.setdefault("ts", now_iso())
    message = _serialize(record)
    if LOCAL_LOG_FILE:
        try:
            with open(LOCAL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"[rag_logging] local log write failed ({e})", file=sys.stderr)
    if _sink:
        _sink.submit(message)


def describe_chunks(scored_docs):
    """Turn [(Document, score), ...] from FAISS into the JSON-safe chunk list we log."""
    chunks = []
    for rank, (doc, score) in enumerate(scored_docs, start=1):
        text, truncated = _truncate(doc.page_content, CHUNK_CHARS)
        metadata = doc.metadata or {}
        chunks.append({
            "rank": rank,
            "chunk_id": chunk_id(doc.page_content),
            # Cosine similarity: higher = more similar. See cosine_from_l2_sq().
            "score": round(cosine_from_l2_sq(score), 6),
            # The raw squared L2 distance FAISS returned, kept so the log stays faithful
            # to what the index actually computed.
            "l2_sq": round(float(score), 6),
            "page": metadata.get("page"),
            "source": metadata.get("source"),
            "chars": len(doc.page_content),
            "text": text,
            "text_truncated": truncated,
        })
    return chunks


def truncate_answer(answer):
    return _truncate(answer, ANSWER_CHARS)