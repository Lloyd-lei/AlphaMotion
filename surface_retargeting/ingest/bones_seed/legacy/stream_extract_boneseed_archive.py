#!/usr/bin/env python3
"""Stream a BONES-SEED tar.gz from Hugging Face directly into tar extraction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import get_token


REPO_ID = "bones-studio/seed"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def bytes_gb(n: int | float) -> float:
    return round(float(n) / 1_000_000_000, 3)


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def disk_report(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": bytes_gb(usage.total),
        "used_gb": bytes_gb(usage.used),
        "free_gb": bytes_gb(usage.free),
    }


def write_status(status_path: Path, data: dict[str, object]) -> None:
    data["updated_at"] = now()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def load_status(status_path: Path, out: Path) -> dict[str, object]:
    if status_path.exists():
        data = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        data = {"repo": REPO_ID, "archives": {}}
    data.setdefault("started_at", now())
    data["repo"] = REPO_ID
    data["data_dir"] = str(out)
    data.setdefault("archives", {})
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", type=Path, default=Path("alpha_motion2/wbc_expert/data/raw/bones_seed_full"))
    parser.add_argument("--status", type=Path, default=Path("alpha_motion2/wbc_expert/runs/boneseed_sync_status.json"))
    parser.add_argument("--expected-size-gb", type=float, default=45.468)
    parser.add_argument("--expected-files", type=int, default=142220)
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    status_path = args.status.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    extracted_name = args.archive.removesuffix(".tar.gz")
    extracted_dir = out / extracted_name
    done_marker = out / f".{args.archive}.extracted"
    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token available; run `hf auth login` first.")

    status = load_status(status_path, out)
    archives = dict(status.get("archives", {}))
    archives[args.archive] = {
        "state": "stream_extracting",
        "expected_size_gb": args.expected_size_gb,
        "expected_files": args.expected_files,
        "path": str(out / args.archive),
        "streaming": True,
    }
    status.update(
        state="running",
        current_stage=f"stream extract {args.archive}",
        archives=archives,
        disk=disk_report(out),
        extracted_size_gb=bytes_gb(tree_size(out)),
        message=f"streaming {args.archive} from Hugging Face directly into tar",
    )
    write_status(status_path, status)

    if done_marker.exists():
        archives[args.archive]["state"] = "already_extracted"
        status.update(
            state="partial_waiting_verification",
            current_stage=None,
            archives=archives,
            disk=disk_report(out),
            extracted_size_gb=bytes_gb(tree_size(out)),
            message=f"{args.archive} already extracted",
        )
        write_status(status_path, status)
        return

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    cfg_path = out / ".cache" / "curl" / f"{args.archive}.stream.conf"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "\n".join(
            [
                f'url = "{hf_hub_url(REPO_ID, args.archive, repo_type="dataset")}"',
                "location",
                "fail",
                "retry = 100",
                "retry-delay = 10",
                "retry-all-errors",
                f'header = "Authorization: Bearer {token}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(cfg_path, 0o600)

    curl = subprocess.Popen(["curl", "--config", str(cfg_path)], stdout=subprocess.PIPE)
    assert curl.stdout is not None
    tar = subprocess.Popen(["tar", "-xzf", "-", "-C", str(out)], stdin=curl.stdout)
    curl.stdout.close()
    tar_rc = tar.wait()
    curl_rc = curl.wait()
    if curl_rc != 0 or tar_rc != 0:
        archives[args.archive]["state"] = "failed"
        status.update(
            state="failed",
            current_stage=f"stream extract {args.archive}",
            archives=archives,
            disk=disk_report(out),
            extracted_size_gb=bytes_gb(tree_size(out)),
            message=f"stream extract failed: curl={curl_rc} tar={tar_rc}",
        )
        write_status(status_path, status)
        raise RuntimeError(f"stream extract failed: curl={curl_rc} tar={tar_rc}")

    file_count = 0
    if extracted_dir.exists():
        file_count = sum(1 for p in extracted_dir.rglob("*") if p.is_file())
    done_marker.write_text(now(), encoding="utf-8")
    archives[args.archive].update(
        state="extracted",
        file_count=file_count,
        archive_deleted=True,
    )
    status.update(
        state="complete",
        current_stage=None,
        archives=archives,
        disk=disk_report(out),
        extracted_size_gb=bytes_gb(tree_size(out)),
        message=f"{args.archive} stream extraction complete",
    )
    write_status(status_path, status)


if __name__ == "__main__":
    main()
