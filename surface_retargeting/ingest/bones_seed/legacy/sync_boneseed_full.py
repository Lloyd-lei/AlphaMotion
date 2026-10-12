#!/usr/bin/env python3
"""Download and extract BONES-SEED in resumable stages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import hf_hub_url, snapshot_download
from huggingface_hub.utils import get_token


REPO_ID = "bones-studio/seed"
ARCHIVES = [
    ("g1.tar.gz", 23_499_973_647),
    ("soma_uniform.tar.gz", 45_188_088_297),
    ("soma_proportional.tar.gz", 45_468_083_856),
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def bytes_gb(n: int | float) -> float:
    return round(float(n) / 1_000_000_000, 3)


def disk_report(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total_gb": bytes_gb(usage.total),
        "used_gb": bytes_gb(usage.used),
        "free_gb": bytes_gb(usage.free),
    }


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


class Status:
    def __init__(self, path: Path, data_dir: Path):
        self.path = path
        self.data_dir = data_dir
        self.data: dict[str, object] = {
            "started_at": now(),
            "updated_at": now(),
            "state": "starting",
            "repo": REPO_ID,
            "data_dir": str(data_dir),
            "current_stage": None,
            "archives": {},
            "disk": disk_report(data_dir.parent),
            "extracted_size_gb": 0.0,
            "message": "",
        }
        self.write()

    def update(self, **kwargs: object) -> None:
        self.data.update(kwargs)
        self.data["updated_at"] = now()
        self.data["disk"] = disk_report(self.data_dir.parent)
        self.data["extracted_size_gb"] = bytes_gb(tree_size(self.data_dir))
        self.write()

    def archive(self, filename: str, **kwargs: object) -> None:
        archives = dict(self.data.get("archives", {}))
        item = dict(archives.get(filename, {}))
        item.update(kwargs)
        archives[filename] = item
        self.update(archives=archives)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def run(cmd: list[str], cwd: Path, status: Status, stage: str) -> None:
    status.update(state="running", current_stage=stage, message=" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_resumable_download(
    cmd: list[str],
    cwd: Path,
    status: Status,
    stage: str,
    max_attempts: int,
    retry_sleep: float,
) -> None:
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        status.update(
            state="running",
            current_stage=stage,
            message=f"attempt {attempt}/{max_attempts}: {' '.join(cmd)}",
        )
        try:
            subprocess.run(cmd, cwd=str(cwd), check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            status.update(
                state="running",
                current_stage=stage,
                message=f"download attempt {attempt}/{max_attempts} failed with code {exc.returncode}; retrying in {retry_sleep:g}s",
            )
            time.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


def download_archive(filename: str, out: Path, status: Status, max_attempts: int, retry_sleep: float) -> Path:
    archive_path = out / filename
    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token available; run `hf auth login` first.")

    aria2c = shutil.which("aria2c")
    if aria2c:
        cfg_path = out / ".cache" / "aria2" / f"{filename}.conf"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            "\n".join(
                [
                    f"dir={out}",
                    f"out={filename}",
                    "continue=true",
                    "allow-overwrite=true",
                    "auto-file-renaming=false",
                    "file-allocation=none",
                    "split=8",
                    "max-connection-per-server=8",
                    "min-split-size=16M",
                    "retry-wait=10",
                    "max-tries=0",
                    "timeout=60",
                    "connect-timeout=30",
                    "summary-interval=30",
                    "console-log-level=notice",
                    f"header=Authorization: Bearer {token}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(cfg_path, 0o600)
        url = hf_hub_url(REPO_ID, filename, repo_type="dataset")
        run_resumable_download(
            [aria2c, f"--conf-path={cfg_path}", url],
            cwd=out,
            status=status,
            stage=f"download {filename}",
            max_attempts=max_attempts,
            retry_sleep=retry_sleep,
        )
    else:
        # Fallback to curl with a private config file so the token is not exposed
        # in process arguments.
        cfg_path = out / ".cache" / "curl" / f"{filename}.conf"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            "\n".join(
                [
                    f'url = "{hf_hub_url(REPO_ID, filename, repo_type="dataset")}"',
                    f'output = "{archive_path}"',
                    "location",
                    "continue-at = -",
                    "retry = 100",
                    "retry-delay = 10",
                    f'header = "Authorization: Bearer {token}"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(cfg_path, 0o600)
        run_resumable_download(
            ["curl", "--config", str(cfg_path)],
            cwd=out,
            status=status,
            stage=f"download {filename}",
            max_attempts=max_attempts,
            retry_sleep=retry_sleep,
        )

    if not archive_path.exists():
        raise RuntimeError(f"Download finished but file is missing: {archive_path}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("alpha_motion2/wbc_expert/data/raw/bones_seed_full"))
    parser.add_argument("--status", type=Path, default=Path("alpha_motion2/wbc_expert/runs/boneseed_sync_status.json"))
    parser.add_argument("--delete-archives", action="store_true", help="Delete each .tar.gz after extraction to save disk.")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--download-max-attempts", type=int, default=1000)
    parser.add_argument("--download-retry-sleep", type=float, default=60.0)
    parser.add_argument(
        "--archives",
        nargs="*",
        choices=[name for name, _ in ARCHIVES],
        default=[name for name, _ in ARCHIVES],
        help="Subset of large motion archives to download/extract.",
    )
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = Status(args.status.expanduser().resolve(), out)
    status.update(state="running", message="downloading metadata and soma shape files")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=out,
            allow_patterns=["README.md", "LICENSE.md", "metadata/*", "soma_shapes/*"],
        )
        status.update(message="metadata and soma shape files ready")

        requested = set(args.archives)
        for filename, expected_size in ARCHIVES:
            if filename not in requested:
                status.archive(filename, state="skipped", expected_size_gb=bytes_gb(expected_size))
                continue
            archive_path = out / filename
            done_marker = out / f".{filename}.extracted"
            if done_marker.exists():
                status.archive(filename, state="already_extracted", expected_size_gb=bytes_gb(expected_size))
                continue

            free_gb = disk_report(out)["free_gb"]
            if free_gb < bytes_gb(expected_size) + args.min_free_gb:
                raise RuntimeError(f"not enough disk before downloading {filename}: free={free_gb}GB")

            status.archive(filename, state="downloading", expected_size_gb=bytes_gb(expected_size), path=str(archive_path))
            archive_path = download_archive(
                filename,
                out,
                status,
                max_attempts=args.download_max_attempts,
                retry_sleep=args.download_retry_sleep,
            )
            actual_size = archive_path.stat().st_size
            if actual_size < int(expected_size * 0.99):
                raise RuntimeError(f"Downloaded {filename} is too small: {actual_size} < {expected_size}")
            status.archive(filename, state="downloaded", path=str(archive_path), size_gb=bytes_gb(actual_size))

            status.archive(filename, state="extracting")
            # GNU tar is much faster than Python tarfile for these large gzip archives.
            run(["tar", "-xzf", str(archive_path), "-C", str(out)], cwd=out, status=status, stage=f"extract {filename}")
            done_marker.write_text(now(), encoding="utf-8")
            status.archive(filename, state="extracted")

            if args.delete_archives:
                archive_path.unlink(missing_ok=True)
                status.archive(filename, archive_deleted=True)

        status.update(state="complete", current_stage=None, message="BONES-SEED download and extraction complete")
    except Exception as exc:
        status.update(state="failed", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
