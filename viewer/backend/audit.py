"""Append-only human audit interface for SCHEMA.md v1 (stdlib only).

The catalog supplied by the application is trusted server metadata, not form data.
No constructor or read operation creates files. Only append operations write logs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Callable, Iterable, Mapping

UI_VERSION = "1.0.0"
REASONS = (
    "penetration_ground", "penetration_object", "self_collision", "hover",
    "slip", "contact_mismatch_human", "object_misaligned", "jitter",
    "unnatural_pose", "retarget_failure", "source_bad", "floor_wrong",
    "label_wrong", "other",
)
VERDICTS = ("accept", "reject", "needs_fix")
ISSUE_SCOPES = ("robot", "source", "engine")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)$")
MAX_LINE_BYTES = 1_048_576


class AuditError(ValueError):
    code = "audit_error"

    def as_dict(self):
        return {"code": self.code, "message": str(self)}


class AuditValidationError(AuditError):
    code = "invalid_audit_record"


class AuditLogError(AuditError):
    code = "invalid_existing_audit_log"


class StaleSnapshotError(AuditError):
    code = "stale_snapshot"

    def __init__(self, message, *, committed=False):
        super().__init__(message)
        self.committed = committed

    def as_dict(self):
        return {**super().as_dict(), "committed": self.committed,
                "reload_required": True}


def _string(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise AuditValidationError(f"{name}: expected {'possibly empty ' if empty else 'non-empty '}string")
    if "\x00" in value:
        raise AuditValidationError(f"{name}: NUL is not allowed")


def _hash(value, name="npz_sha256"):
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise AuditValidationError(f"{name}: expected 64 hexadecimal SHA-256 characters")
    return value.lower()


def _utc(value):
    if not isinstance(value, str) or not UTC_RE.fullmatch(value):
        raise AuditValidationError("ts: expected UTC ISO-8601 timestamp ending Z or +00:00")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditValidationError(f"ts: invalid calendar timestamp ({exc})") from exc


def _reasons(value):
    if not isinstance(value, list) or any(not isinstance(x, str) or x not in REASONS for x in value):
        raise AuditValidationError("reasons: use only the explicit SCHEMA.md reason table; 'penetration' is not a valid reason")
    if len(set(value)) != len(value):
        raise AuditValidationError("reasons: duplicate entries")


def _keys(record, required, optional=()):
    if not isinstance(record, dict):
        raise AuditValidationError("record: expected JSON object")
    missing = sorted(set(required) - record.keys())
    extra = sorted(record.keys() - set(required) - set(optional))
    if missing:
        raise AuditValidationError("missing fields: " + ", ".join(missing))
    if extra:
        raise AuditValidationError("unknown fields: " + ", ".join(extra))


def validate_decision(record, *, frames=None):
    """Validate exact v1 shape. Return a detached normalized JSON-compatible dict."""
    _keys(record, ("item_id", "npz_sha256", "verdict", "reasons", "note", "reviewer", "ts", "ui_version"), ("frame_ranges",))
    for key in ("item_id", "reviewer", "ui_version"):
        _string(record[key], key)
    _string(record["note"], "note", empty=True)
    digest = _hash(record["npz_sha256"])
    if record["verdict"] not in VERDICTS:
        raise AuditValidationError("verdict: expected accept, reject or needs_fix")
    _reasons(record["reasons"])
    _utc(record["ts"])
    if frames is not None and (type(frames) is not int or frames < 1):
        raise AuditValidationError("frames: metadata must be a positive integer")
    ranges = record.get("frame_ranges", [])
    if not isinstance(ranges, list):
        raise AuditValidationError("frame_ranges: expected list of half-open [start, end] ranges")
    for i, pair in enumerate(ranges):
        if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair):
            raise AuditValidationError(f"frame_ranges[{i}]: expected two integer indices")
        start, end = pair
        if start < 0 or end <= start:
            raise AuditValidationError(f"frame_ranges[{i}]: require 0 <= start < end (end exclusive)")
        if frames is not None and end > frames:
            raise AuditValidationError(f"frame_ranges[{i}]: end {end} exceeds frame count {frames}")
    result = dict(record)
    result.update(npz_sha256=digest, reasons=list(record["reasons"]), frame_ranges=[list(p) for p in ranges])
    _encode(result)
    return result


def validate_issue(record):
    _keys(record, ("scope", "target", "reasons", "note", "example_items", "reviewer", "ts"))
    if record["scope"] not in ISSUE_SCOPES:
        raise AuditValidationError("scope: expected robot, source or engine")
    for key in ("target", "reviewer"):
        _string(record[key], key)
    _string(record["note"], "note", empty=True)
    _reasons(record["reasons"])
    _utc(record["ts"])
    if not isinstance(record["example_items"], list):
        raise AuditValidationError("example_items: expected list of item IDs")
    for item in record["example_items"]:
        _string(item, "example_items[]")
    if len(set(record["example_items"])) != len(record["example_items"]):
        raise AuditValidationError("example_items: duplicate item IDs")
    result = dict(record, reasons=list(record["reasons"]), example_items=list(record["example_items"]))
    _encode(result)
    return result


def _encode(record):
    try:
        line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AuditValidationError(f"record is not valid UTF-8 JSON: {exc}") from exc
    if len(line) > MAX_LINE_BYTES:
        raise AuditValidationError(f"record exceeds {MAX_LINE_BYTES} bytes")
    return line


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise AuditValidationError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def decode_line(line):
    try:
        return json.loads(line, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(AuditValidationError(f"non-finite JSON number: {value}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditValidationError(f"invalid JSON: {exc}") from exc


def _bounded_lines(stream):
    """Keep malformed oversized lines from allocating unbounded memory."""
    number = 0
    while True:
        raw = stream.readline(MAX_LINE_BYTES + 1)
        if not raw:
            return
        number += 1
        oversized = len(raw) > MAX_LINE_BYTES
        if oversized and not raw.endswith(b"\n"):
            while True:
                rest = stream.readline(MAX_LINE_BYTES + 1)
                if not rest or rest.endswith(b"\n"):
                    break
        yield number, raw, oversized


def _safe_directory(path):
    """Open each directory component without following symlinks, including parents."""
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            new = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd
    except Exception:
        os.close(fd)
        raise


def sha256_file(path, *, allowed_root=None):
    """Hash a regular file without following links; reject concurrent replacement/write.

    Used by the CLI or application's no-argument current_hash_callback. Hashing is
    bounded to a 1 MiB buffer. A changed inode/size/mtime/ctime is not a snapshot.
    """
    path = Path(os.path.abspath(path))
    if allowed_root is not None:
        root = Path(os.path.abspath(allowed_root))
        if not path.is_relative_to(root):
            raise AuditValidationError("npz path escapes allowed data root")
    dfd = _safe_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AuditValidationError("npz path is not a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        current = os.stat(path.name, dir_fd=dfd, follow_symlinks=False)
        def identity(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise StaleSnapshotError("数据读取期间已变化，请重新加载后审阅")
        return digest.hexdigest()
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dfd)


def inspect_jsonl(path, *, kind="decision", frames_by_item=None):
    """Read-only validation; collect exact line errors instead of repairing logs."""
    path = Path(path)
    validator = validate_decision if kind == "decision" else validate_issue
    records, errors = [], []
    dfd = _safe_directory(path.parent)
    try:
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        except FileNotFoundError:
            return {"records": [], "errors": [], "exists": False}
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AuditLogError(f"{path}: not a regular file")
            fcntl.flock(fd, fcntl.LOCK_SH)
            with os.fdopen(os.dup(fd), "rb") as stream:
                for line_number, raw, oversized in _bounded_lines(stream):
                    try:
                        if oversized:
                            raise AuditValidationError("line exceeds record size limit")
                        if not raw.endswith(b"\n"):
                            raise AuditValidationError("truncated log: line has no terminating newline")
                        value = decode_line(raw)
                        frames = (frames_by_item or {}).get(value.get("item_id")) if isinstance(value, dict) else None
                        result = validator(value, frames=frames) if kind == "decision" else validator(value)
                        records.append({"line": line_number, "record": result})
                    except AuditValidationError as exc:
                        errors.append({"line": line_number, "message": str(exc)})
        finally:
            os.close(fd)
    finally:
        os.close(dfd)
    return {"records": records, "errors": errors, "exists": True}


class AuditStore:
    def __init__(self, audit_root, *, allowed_data_root=None, ui_version=UI_VERSION):
        self.audit_root = Path(os.path.abspath(audit_root))
        self.allowed_data_root = Path(os.path.abspath(allowed_data_root)) if allowed_data_root else None
        self.ui_version = ui_version
        # Validation is read-only; the factory owns creating the audit directory.
        fd = _safe_directory(self.audit_root)
        os.close(fd)

    def _read(self, name, kind):
        result = inspect_jsonl(self.audit_root / name, kind=kind)
        if result["errors"]:
            detail = "; ".join(f"line {x['line']}: {x['message']}" for x in result["errors"])
            raise AuditLogError(f"{name}: {detail}")
        return [x["record"] for x in result["records"]]

    def read_decisions(self):
        return self._read("decisions.jsonl", "decision")

    def read_issues(self):
        return self._read("issues.jsonl", "issue")

    def read_records(self):
        return {"decisions": self.read_decisions(), "issues": self.read_issues()}

    def append_decision(self, payload, item_metadata, current_hash_callback: Callable[[], str]):
        """Require submitted hash == displayed snapshot hash == freshly hashed source.

        Hash callback runs under the append lock twice before the single write and
        once after fsync. A post-write external source change is reported as stale
        with committed=True; an append-only log must never be rolled back.
        """
        if not isinstance(item_metadata, Mapping):
            raise AuditValidationError("item_metadata: expected server catalog metadata")
        record = validate_decision(payload, frames=item_metadata.get("frames"))
        if item_metadata.get("item_id") != record["item_id"]:
            raise AuditValidationError("item_id does not match the viewed item")
        if item_metadata.get("frames") is None:
            raise AuditValidationError("frames: viewed item metadata is required")
        snapshot = _hash(item_metadata.get("npz_sha256"), "viewed npz_sha256")
        if record["npz_sha256"] != snapshot:
            raise StaleSnapshotError("提交哈希与正在审阅的快照不一致，请重新加载")
        if not callable(current_hash_callback):
            raise AuditValidationError("current_hash_callback is required")
        if self.allowed_data_root is not None and item_metadata.get("npz"):
            path = Path(os.path.abspath(item_metadata["npz"]))
            if not path.is_relative_to(self.allowed_data_root):
                raise AuditValidationError("npz path escapes allowed data root")
        def check(committed=False):
            try:
                current = _hash(current_hash_callback(), "current file SHA-256")
            except StaleSnapshotError as exc:
                raise StaleSnapshotError(str(exc), committed=committed) from exc
            except (OSError, AuditValidationError) as exc:
                raise StaleSnapshotError(f"无法核验当前文件，请重新加载：{exc}", committed=committed) from exc
            if current != snapshot:
                raise StaleSnapshotError("文件已重新生成，当前判决已过期，需要重新审阅", committed=committed)
        return self._append("decisions.jsonl", record, validate_decision, check)

    def append_issue(self, payload):
        return self._append("issues.jsonl", validate_issue(payload), validate_issue)

    def _append(self, name, record, validator, check=None):
        line = _encode(record)
        dfd = _safe_directory(self.audit_root)
        fd = None
        try:
            flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW
            try:
                fd = os.open(name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dfd)
            except FileExistsError:
                fd = os.open(name, flags, dir_fd=dfd)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AuditLogError(f"{name}: not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            # Refuse damaged history; never truncate or silently discard a bad line.
            with os.fdopen(os.dup(fd), "rb") as stream:
                stream.seek(0)
                for i, raw, oversized in _bounded_lines(stream):
                    try:
                        if oversized or not raw.endswith(b"\n"):
                            raise AuditValidationError("oversized or unterminated line")
                        validator(decode_line(raw))
                    except AuditValidationError as exc:
                        raise AuditLogError(f"{name}: line {i}: {exc}; log unchanged") from exc
            if check:
                check()
            before = os.fstat(fd)
            linked = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (linked.st_dev, linked.st_ino):
                raise AuditLogError(f"{name}: replaced while acquiring append lock")
            if check:
                check()
            # Exactly one O_APPEND write. If the OS reports a partial write, keep
            # the evidence and fail loudly; do not repair an append-only history.
            written = os.write(fd, line)
            os.fsync(fd)
            os.fsync(dfd)
            if written != len(line):
                raise AuditLogError(f"{name}: partial write ({written}/{len(line)}); inspect log before further writes")
            if check:
                check(committed=True)
            return record
        finally:
            if fd is not None:
                os.close(fd)
            os.close(dfd)

    def item_statuses(self, items):
        latest = {}
        for record in self.read_decisions():
            latest[record["item_id"]] = record  # physical append order, not user clock
        result = []
        for item in items:
            record = latest.get(item["item_id"])
            current = item.get("npz_sha256")
            known = isinstance(current, str) and bool(SHA256_RE.fullmatch(current))
            stale = bool(record and (not known or record["npz_sha256"] != current.lower()))
            result.append({"item_id": item["item_id"], "decision": record,
                           "status": "stale" if stale else ("reviewed" if record else "pending"),
                           "needs_review": record is None or stale,
                           "current_hash_known": known,
                           "partial": bool(record and record["frame_ranges"])})
        return result

    def analytics(self, items):
        return build_analytics(items, self.read_decisions())


def wilson_interval(successes, n, *, z=1.959963984540054):
    if type(n) is not int or type(successes) is not int or not 0 <= successes <= n:
        raise AuditValidationError("Wilson counts require integers with 0 <= successes <= n")
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _group_summary(rows):
    current = [r for r in rows if r["status"] == "reviewed"]
    whole = [r for r in current if not r["decision"]["frame_ranges"]]
    accepted = sum(r["decision"]["verdict"] == "accept" for r in whole)
    verdicts = Counter(r["decision"]["verdict"] for r in current)
    return {"total": len(rows), "reviewed": len(current),
            "pending": len(rows) - len(current), "stale": sum(r["status"] == "stale" for r in rows),
            "partial_decisions": len(current) - len(whole), "whole_item_reviewed": len(whole),
            "accepted": accepted, "accept_rate": accepted / len(whole) if whole else None,
            "verdict_counts": {k: verdicts[k] for k in VERDICTS},
            "accept_rate_denominator": "current-hash latest whole-item decisions only"}


def build_analytics(items: Iterable[Mapping], decisions: Iterable[Mapping]):
    """Latest current-hash decisions; risk-pool shuffle never establishes randomness.

    `queued=True` AND original `sample_kind=random` are both required for the
    confidence interval. Population, survey design and completeness remain unknown
    unless declared by the upstream queue producer; the UI must expose this basis.
    """
    catalog = {item["item_id"]: item for item in items}
    latest = {}
    for record in decisions:
        latest[record["item_id"]] = validate_decision(dict(record))
    rows = []
    for key, item in catalog.items():
        record = latest.get(key)
        digest = item.get("npz_sha256")
        known = isinstance(digest, str) and bool(SHA256_RE.fullmatch(digest))
        stale = bool(record and (not known or record["npz_sha256"] != digest.lower()))
        status = "stale" if stale else ("reviewed" if record else "pending")
        rows.append({"item": item, "decision": record, "status": status})
    by_robot, by_source = defaultdict(list), defaultdict(list)
    reasons = Counter()
    for row in rows:
        by_robot[str(row["item"].get("robot") or "unknown")].append(row)
        by_source[str(row["item"].get("source") or "unknown")].append(row)
        if row["status"] == "reviewed" and row["decision"]["verdict"] in ("reject", "needs_fix"):
            reasons.update(row["decision"]["reasons"])
    random_rows = [r for r in rows if r["item"].get("queued") is True and r["item"].get("sample_kind") == "random"]
    random_summary = _group_summary(random_rows)
    n = random_summary["whole_item_reviewed"]
    successes = random_summary["accepted"]
    random_summary.update({"n": n, "successes": successes, "confidence": 0.95,
                           "wilson_95": wilson_interval(successes, n),
                           "available": n > 0,
                           "basis": "queue.jsonl sample_kind=random and queued=true; browse ordering never changes cohort membership",
                           "population": "upstream sampling population not independently established",
                           "caveat": "仅已完成且哈希有效的整条人工判决；未审样本和帧段判决未计入，未证明全量总体无偏。"})
    by_item = {}
    for row in rows:
        record = row["decision"]
        by_item[row["item"]["item_id"]] = {
            "status": row["status"], "needs_review": row["status"] != "reviewed",
            "stale": row["status"] == "stale", "partial": bool(record and record["frame_ranges"]),
            "verdict": record["verdict"] if record else None, "decision": record,
        }
    return {**_group_summary(rows), "by_item": by_item,
            "by_robot": {k: _group_summary(v) for k, v in sorted(by_robot.items())},
            "by_source": {k: _group_summary(v) for k, v in sorted(by_source.items())},
            "reason_ranking": [{"reason": k, "count": v} for k, v in sorted(reasons.items(), key=lambda p: (-p[1], p[0]))],
            "random_sample": random_summary,
            "orphan_decision_items": sorted(set(latest) - set(catalog)),
            "status_basis": "current server catalog hashes; latest append wins; stale is pending re-review"}
