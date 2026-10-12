"""All mutations use temporary fixtures below this project's tests directory."""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from backend.audit import (
    AuditLogError, AuditStore, AuditValidationError, REASONS, StaleSnapshotError,
    build_analytics, decode_line, inspect_jsonl, sha256_file, validate_decision,
    validate_issue, wilson_interval,
)

PROJECT = Path(__file__).resolve().parents[1]
HASH = hashlib.sha256(b"fixture").hexdigest()


def decision(item_id="clip::robot", digest=HASH, **updates):
    result = {"item_id": item_id, "npz_sha256": digest, "verdict": "accept",
              "frame_ranges": [], "reasons": [], "note": "人工隔离测试样本",
              "reviewer": "fixture_only", "ts": "2026-10-05T08:00:00Z", "ui_version": "1.0.0"}
    result.update(updates)
    return result


def issue(**updates):
    result = {"scope": "robot", "target": "robot", "reasons": ["self_collision"],
              "note": "isolated test only", "example_items": ["clip::robot"],
              "reviewer": "fixture_only", "ts": "2026-10-05T08:00:00+00:00"}
    result.update(updates)
    return result


def concurrent_writer(root, worker, count):
    store = AuditStore(root)
    for n in range(count):
        item_id = f"worker{worker}/clip{n}::robot"
        store.append_decision(decision(item_id), {"item_id": item_id, "npz_sha256": HASH, "frames": 40}, lambda: HASH)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".audit-fixture-", dir=PROJECT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.source = self.data / "clip.npz"
        self.source.write_bytes(b"fixture")
        self.logs = self.root / "audit"
        self.logs.mkdir()
        self.store = AuditStore(self.logs, allowed_data_root=self.data)
        self.metadata = {"item_id": "clip::robot", "npz_sha256": HASH, "npz": str(self.source), "frames": 40}

    def hash_current(self):
        return sha256_file(self.source, allowed_root=self.data)

    def append(self, payload=None, callback=None):
        return self.store.append_decision(payload or decision(), self.metadata, callback or self.hash_current)

    def test_exact_reason_table_and_verdicts(self):
        for reason in REASONS:
            self.assertEqual(validate_decision(decision(reasons=[reason]))["reasons"], [reason])
        for invalid in ["penetration", "ground_penetration", 1, None]:
            with self.subTest(reason=invalid), self.assertRaises(AuditValidationError):
                validate_decision(decision(reasons=[invalid]))
        for invalid in ["pass", "fix", None, []]:
            with self.subTest(verdict=invalid), self.assertRaises(AuditValidationError):
                validate_decision(decision(verdict=invalid))

    def test_required_fields_and_utc(self):
        for key in ["item_id", "npz_sha256", "verdict", "reasons", "note", "reviewer", "ts", "ui_version"]:
            value = decision()
            del value[key]
            with self.subTest(key=key), self.assertRaises(AuditValidationError):
                validate_decision(value)
        for ts in ["2026-10-05", "2026-10-05T08:00:00", "2026-10-05T08:00:00-04:00", "2026-02-30T08:00:00Z"]:
            with self.subTest(ts=ts), self.assertRaises(AuditValidationError):
                validate_decision(decision(ts=ts))
        with self.assertRaises(AuditValidationError):
            validate_decision(decision(sample_kind="random"))

    def test_hash_validation(self):
        for digest in ["", "f" * 63, "g" * 64, None, 123]:
            with self.subTest(hash=digest), self.assertRaises(AuditValidationError):
                validate_decision(decision(digest=digest))
        self.assertEqual(validate_decision(decision(digest=HASH.upper()))["npz_sha256"], HASH)

    def test_half_open_ranges_and_frame_limits(self):
        self.assertEqual(validate_decision(decision(frame_ranges=[[0, 40], [3, 4]]), frames=40)["frame_ranges"], [[0, 40], [3, 4]])
        value = decision()
        del value["frame_ranges"]
        self.assertEqual(validate_decision(value)["frame_ranges"], [])
        for ranges in [[[1, 1]], [[-1, 3]], [[3, 2]], [[0, 41]], [[False, 3]], [[0.0, 3]], [[0, 2, 3]], "0:4"]:
            with self.subTest(ranges=ranges), self.assertRaises(AuditValidationError):
                validate_decision(decision(frame_ranges=ranges), frames=40)

    def test_append_preserves_exact_history_bytes_and_single_write(self):
        historical = json.dumps(decision(note="retain original spaces"), indent=None, ensure_ascii=False).encode() + b"\n"
        log = self.logs / "decisions.jsonl"
        log.write_bytes(historical)
        original_write = os.write
        with mock.patch("backend.audit.os.write", wraps=original_write) as writer:
            returned = self.append(decision(verdict="needs_fix", reasons=["jitter"]))
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(returned["verdict"], "needs_fix")
        self.assertTrue(log.read_bytes().startswith(historical))
        self.assertEqual(len(self.store.read_decisions()), 2)

    def test_malformed_or_truncated_history_is_not_repaired(self):
        log = self.logs / "decisions.jsonl"
        for damaged in [b'{"item_id": "broken"}\n', json.dumps(decision()).encode(), b"\n", b"{broken}\n"]:
            log.write_bytes(damaged)
            with self.subTest(damaged=damaged[:40]), self.assertRaises(AuditLogError):
                self.append()
            self.assertEqual(log.read_bytes(), damaged)
            with self.assertRaises(AuditLogError):
                self.store.read_decisions()

    def test_existing_log_inspection_reports_exact_lines(self):
        log = self.logs / "decisions.jsonl"
        log.write_bytes(json.dumps(decision()).encode() + b"\n{bad}\n" + json.dumps(decision()).encode())
        inspected = inspect_jsonl(log)
        self.assertEqual([x["line"] for x in inspected["errors"]], [2, 3])
        self.assertEqual(len(inspected["records"]), 1)

    def test_duplicate_json_keys_and_nan_are_rejected(self):
        for raw in [b'{"item_id":"x","item_id":"y"}', b'{"value":NaN}', b'{"value":Infinity}']:
            with self.subTest(raw=raw), self.assertRaises(AuditValidationError):
                decode_line(raw)

    def test_changed_before_submit_rejected_without_decision(self):
        self.source.write_bytes(b"new-generation")
        with self.assertRaises(StaleSnapshotError) as caught:
            self.append()
        self.assertFalse(caught.exception.committed)
        self.assertEqual(self.store.read_decisions(), [])

    def test_request_hash_must_match_viewed_snapshot(self):
        with self.assertRaises(StaleSnapshotError):
            self.append(decision(digest="a" * 64))
        with self.assertRaises(AuditValidationError):
            self.append(decision(item_id="other::robot"))
        self.assertEqual(self.store.read_decisions(), [])

    def test_change_between_precommit_hashes_is_rejected(self):
        calls = 0
        def callback():
            nonlocal calls
            calls += 1
            current = self.hash_current()
            if calls == 1:
                self.source.write_bytes(b"changed-during-submit")
            return current
        with self.assertRaises(StaleSnapshotError) as caught:
            self.append(callback=callback)
        self.assertFalse(caught.exception.committed)
        self.assertEqual(calls, 2)
        self.assertEqual(self.store.read_decisions(), [])

    def test_change_after_append_reported_committed_stale_without_rewrite(self):
        original_write = os.write
        def changing_write(fd, content):
            result = original_write(fd, content)
            self.source.write_bytes(b"producer-replaced-file-after-append")
            return result
        with mock.patch("backend.audit.os.write", side_effect=changing_write):
            with self.assertRaises(StaleSnapshotError) as caught:
                self.append()
        self.assertTrue(caught.exception.committed)
        self.assertEqual(self.store.read_decisions()[0]["npz_sha256"], HASH)
        current_item = dict(self.metadata, npz_sha256=self.hash_current())
        self.assertEqual(self.store.item_statuses([current_item])[0]["status"], "stale")

    def test_changed_during_hash_is_rejected(self):
        original_read = os.read
        changed = False
        def changing_read(fd, size):
            nonlocal changed
            value = original_read(fd, size)
            if not changed:
                changed = True
                self.source.write_bytes(b"larger-source-changed")
            return value
        with mock.patch("backend.audit.os.read", side_effect=changing_read):
            with self.assertRaises(StaleSnapshotError):
                self.hash_current()

    def test_symlink_log_parent_and_source_escape_refused(self):
        outside = self.root / "outside.jsonl"
        outside.write_text("untouched", encoding="utf-8")
        (self.logs / "decisions.jsonl").symlink_to(outside)
        with self.assertRaises(OSError):
            self.append()
        self.assertEqual(outside.read_text(), "untouched")
        linked_root = self.root / "linked_audit"
        linked_root.symlink_to(self.logs, target_is_directory=True)
        with self.assertRaises(OSError):
            AuditStore(linked_root)
        link = self.data / "linked.npz"
        link.symlink_to(self.source)
        with self.assertRaises(OSError):
            sha256_file(link, allowed_root=self.data)
        with self.assertRaises(AuditValidationError):
            sha256_file(outside, allowed_root=self.data)

    def test_concurrent_appends_are_complete(self):
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=concurrent_writer, args=(str(self.logs), w, 7)) for w in range(4)]
        for process in processes:
            process.start()
        codes = []
        for process in processes:
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join(5)
                self.fail("isolated append test child failed to finish")
            codes.append(process.exitcode)
        self.assertEqual(codes, [0] * len(processes))
        records = self.store.read_decisions()
        self.assertEqual(len(records), 28)
        self.assertEqual(len({r["item_id"] for r in records}), 28)

    def test_issue_scope_and_append(self):
        for scope in ("robot", "source", "engine"):
            self.store.append_issue(issue(scope=scope))
        self.assertEqual(len(self.store.read_issues()), 3)
        for invalid in [issue(scope="clip"), issue(ui_version="1.0.0"), issue(reasons=["penetration"]), issue(example_items=[""])]:
            with self.subTest(invalid=invalid), self.assertRaises(AuditValidationError):
                validate_issue(invalid)

    def test_latest_physical_line_wins_even_if_timestamp_older(self):
        self.append(decision(verdict="reject", reasons=["hover"], ts="2026-10-05T12:00:00Z"))
        self.append(decision(verdict="accept", ts="2026-10-05T08:00:00Z"))
        stats = self.store.analytics([dict(self.metadata, robot="robot", source="source")])
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["by_item"]["clip::robot"]["verdict"], "accept")
        self.assertEqual(stats["reason_ranking"], [])

    def test_stats_stale_partial_and_random_cohort(self):
        def item(i, **kw):
            return dict(item_id=i, npz_sha256=HASH, robot="r", source="s", queued=True, **kw)
        items = [item("random-pass", sample_kind="random"), item("random-fail", sample_kind="random"),
                 item("risk-pass", sample_kind="risk"), item("partial", sample_kind="random"),
                 item("pending", sample_kind="random"), item("stale", sample_kind="random"),
                 dict(item_id="scan", npz_sha256=HASH, queued=False, sample_kind="random")]
        records = [decision("random-pass"), decision("random-fail", verdict="reject", reasons=["hover"]),
                   decision("risk-pass"), decision("partial", frame_ranges=[[2, 3]]),
                   decision("stale", digest="f" * 64), decision("scan")]
        stats = build_analytics(items, records)
        self.assertEqual(stats["reviewed"], 5)
        self.assertEqual(stats["pending"], 2)
        self.assertEqual(stats["stale"], 1)
        self.assertEqual(stats["partial_decisions"], 1)
        self.assertEqual(stats["random_sample"]["n"], 2)
        self.assertEqual(stats["random_sample"]["successes"], 1)
        self.assertEqual(stats["random_sample"]["accept_rate"], 0.5)
        self.assertAlmostEqual(stats["random_sample"]["wilson_95"][0], 0.0945312057342307)
        self.assertEqual(stats["reason_ranking"], [{"reason": "hover", "count": 1}])
        self.assertTrue(stats["by_item"]["stale"]["needs_review"])

    def test_random_n0_is_unavailable_and_unknown_hash_is_not_current(self):
        stats = build_analytics([{"item_id": "a", "queued": True, "sample_kind": "random"}], [decision("a")])
        self.assertFalse(stats["random_sample"]["available"])
        self.assertIsNone(stats["random_sample"]["wilson_95"])
        self.assertIsNone(stats["random_sample"]["accept_rate"])
        self.assertEqual(stats["stale"], 1)
        self.assertIsNone(wilson_interval(0, 0))

    def test_cli_validates_schema_and_latest_hash_with_line_errors(self):
        self.append()
        catalog = self.root / "catalog.json"
        catalog.write_text(json.dumps({"items": [self.metadata]}), encoding="utf-8")
        args = [sys.executable, str(PROJECT / "check_decisions.py"), str(self.logs / "decisions.jsonl"), "--catalog", str(catalog), "--data-root", str(self.data), "--check-current", "--json"]
        good = subprocess.run(args, text=True, capture_output=True, timeout=20)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertEqual(json.loads(good.stdout)["checked_current_files"], 1)
        self.source.write_bytes(b"current-data-is-different")
        stale = subprocess.run(args, text=True, capture_output=True, timeout=20)
        self.assertEqual(stale.returncode, 1)
        self.assertEqual(json.loads(stale.stdout)["errors"][0]["line"], 1)


if __name__ == "__main__":
    unittest.main()
