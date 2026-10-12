#!/usr/bin/env python3
"""Read-only line validator for SCHEMA.md decisions/issues JSONL.

Optional current-file validation applies to each item's *last* record. Old history
is retained by design, so earlier hashes may legitimately refer to replaced data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from backend.audit import AuditError, AuditValidationError, decode_line, inspect_jsonl, sha256_file


def load_catalog(path):
    """Accept server catalog JSON {items:[]} or an original queue JSONL."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = decode_line(text)
    except AuditValidationError:
        obj = [decode_line(line) for line in text.splitlines() if line.strip()]
    if isinstance(obj, dict):
        obj = obj.get("items", [obj])
    if not isinstance(obj, list) or any(not isinstance(row, dict) or not row.get("item_id") for row in obj):
        raise AuditValidationError("catalog must contain item objects with item_id")
    return {row["item_id"]: row for row in obj}


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读校验 SCHEMA.md v1 人工判决；输出精确行号，不修复文件")
    parser.add_argument("path", nargs="?", default="/home/arenalabs/umr_factory_data/audit/decisions.jsonl")
    parser.add_argument("--kind", choices=("decision", "issue"), default="decision")
    parser.add_argument("--catalog", help="current catalog JSON or original queue JSONL (supplies item frames and paths)")
    parser.add_argument("--check-current", action="store_true", help="hash source npz files for each item's latest decision; requires --catalog")
    parser.add_argument("--data-root", default="/home/arenalabs/umr_factory_data", help="allowlist root for optional source hashing")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.check_current and (not args.catalog or args.kind != "decision"):
        parser.error("--check-current requires --catalog and --kind decision")
    try:
        catalog = load_catalog(args.catalog) if args.catalog else {}
        result = inspect_jsonl(args.path, kind=args.kind, frames_by_item={k: v.get("frames") for k, v in catalog.items()})
        errors = result["errors"]
        if not result["exists"]:
            errors.append({"line": None, "message": "file does not exist"})
        checked_current = 0
        if args.check_current:
            latest = {entry["record"]["item_id"]: entry for entry in result["records"]}
            for item_id, entry in latest.items():
                try:
                    if item_id not in catalog:
                        raise AuditValidationError(f"current catalog has no item {item_id!r}")
                    item = catalog[item_id]
                    if not item.get("npz"):
                        raise AuditValidationError("current catalog has no npz path")
                    actual = sha256_file(item["npz"], allowed_root=args.data_root)
                    checked_current += 1
                    if actual != entry["record"]["npz_sha256"]:
                        raise AuditValidationError("latest decision is stale: current npz SHA-256 differs")
                except (AuditError, OSError) as exc:
                    errors.append({"line": entry["line"], "message": str(exc)})
        summary = {"path": str(Path(args.path).absolute()), "kind": args.kind,
                   "valid_records": len(result["records"]), "errors": errors,
                   "checked_current_files": checked_current, "ok": not errors,
                   "current_check_scope": "latest record per item only" if args.check_current else "not requested"}
        if args.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            for error in errors:
                suffix = str(error["line"]) if error["line"] is not None else "file"
                print(f"{args.path}:{suffix}: {error['message']}", file=sys.stderr)
            print(f"{'PASS' if not errors else 'FAIL'}: {len(result['records'])} valid records; {len(errors)} errors; {checked_current} current files checked")
        return 1 if errors else 0
    except (AuditError, OSError, UnicodeError) as exc:
        if args.as_json:
            print(json.dumps({"path": args.path, "ok": False, "errors": [{"line": None, "message": str(exc)}]}, ensure_ascii=False))
        else:
            print(f"{args.path}:file: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
