"""AlphaMotion MCP: native evidence, shared motion adapters and persistent jobs."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import uuid
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from .common import digest, file_lock, inside, now, read_json, write_json
from .jobs import Jobs
from . import visual


def create_server(config_path, service=None):
    config_path = Path(config_path).resolve()
    settings = read_json(config_path)
    jobs = Jobs(settings, config_path)
    roots = {**settings.get('evidence_roots', {}), 'agent_state': str(jobs.root)}
    lock = threading.RLock()
    holder = [service]

    def data():
        if holder[0] is None:
            from backend.adapters import FactoryService
            holder[0] = FactoryService(Path(settings['data_root']), Path(settings['cache_root']))
        return holder[0]

    mcp = MCPServer('AlphaMotion Toolchain', version='0.1.0', instructions=(
        'Use toolchain_status and workflow_list first. Motion previews are source-hash-bound kinematic evidence, '
        'not dynamics or physical robot validation. Web and MCP share the same adapters. Long jobs require an '
        'explicit plan then submit; query job_status, job_logs and job_artifacts. Never infer training completion '
        'from resource downloads. Agent notes are separate from human decisions. Unavailable workflows report why.'))
    readonly = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    writes = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    @mcp.tool(annotations=readonly)
    def toolchain_status() -> dict[str, Any]:
        """Discover installed capabilities, evidence roots, capacity and workflow readiness."""
        return {'version': '0.1.0', 'data_root_exists': Path(settings['data_root']).exists(),
                'shared_backend': 'backend.adapters.FactoryService / backend.geometry',
                'free_bytes': shutil.disk_usage(jobs.root).free, 'workflows': jobs.workflows(),
                'evidence_roots': roots, 'visual_tools': ['motion_visualize', 'evidence_read'],
                'human_review_log_writes': False, 'new_training_started_by_setup': False}

    @mcp.tool(annotations=readonly)
    def motion_catalog(offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """List actual supported motion items and hashes; never invent missing robots."""
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError('offset >= 0 and limit 1-100 required')
        with lock:
            result = data().catalog()
        return {**result, 'items': result['items'][offset:offset+limit], 'total': len(result['items']),
                'next_offset': offset + limit if offset + limit < len(result['items']) else None}

    @mcp.tool(annotations=readonly)
    def motion_inspect(key: str, expected_sha256: str | None = None) -> dict[str, Any]:
        """Read source metadata, provenance, contact consistency and known limitations."""
        with lock:
            return data().detail(key, expected_sha256)

    @mcp.tool(annotations=readonly)
    def motion_frame_metrics(key: str, sha256: str, start: int, count: int = 1,
                             mode: Literal['corrected', 'raw'] = 'corrected') -> dict[str, Any]:
        """Measure 1-64 frames using all sampled surface points. Raw mode never reuses corrected contact labels as raw evidence."""
        with lock:
            _, sa = data().scene(key, sha256)
            cm, ca = data().chunk(key, start, count, mode, sha256)
            return visual.metrics(sa, ca, cm)

    @mcp.tool(annotations=writes)
    def motion_visualize(key: str, sha256: str, frames: list[int],
                         view: Literal['front', 'side', 'top', 'iso'] = 'iso',
                         mode: Literal['corrected', 'raw', 'compare'] = 'corrected',
                         show_surface: bool = True, show_human: bool = True) -> CallToolResult:
        """Return a native PNG image plus metrics and provenance. Exact frames; compare raw/corrected with one shared world camera. Up to 6 panels."""
        if not 1 <= len(frames) <= (3 if mode == 'compare' else 6):
            raise ValueError('1-3 comparison frames or 1-6 single-mode frames required')
        with lock:
            sm, sa = data().scene(key, sha256)
            panels = []
            for frame in frames:
                for selected_mode in (['raw', 'corrected'] if mode == 'compare' else [mode]):
                    cm, ca = data().chunk(key, frame, 1, selected_mode, sha256)
                    panels.append((sm, sa, cm, ca))
            png, report = visual.render(panels, view, show_surface, show_human)
            # Revalidate after rendering, before publishing the evidence.
            data().detail(key, sha256)
            stem = 'visuals/' + uuid.uuid4().hex
            path = jobs.root / (stem + '.png'); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
            report.update(source_key=key, source_sha256=sha256, created_unix=now(),
                          artifact={'scope': 'agent_state', 'path': stem + '.png', 'sha256': digest(path)})
            write_json(jobs.root / (stem + '.json'), report)
        return CallToolResult(content=[TextContent(text=json.dumps(report, ensure_ascii=False)),
            ImageContent(data=base64.b64encode(png).decode(), mime_type='image/png')])

    @mcp.tool(annotations=readonly)
    def evidence_list(scope: str, directory: str = '', offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Browse one evidence directory, without recursively scanning datasets or weights."""
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError('Invalid pagination')
        path = inside(roots[scope], directory)
        files = sorted(path.iterdir(), key=lambda p: p.name)
        result = []
        for p in files[offset:offset+limit]:
            if not p.resolve().is_relative_to(Path(roots[scope]).resolve()):
                continue
            result.append({'path': str(p.relative_to(roots[scope])), 'directory': p.is_dir(),
                           'bytes': p.stat().st_size if p.is_file() else None})
        return {'scope': scope, 'items': result, 'total': len(files),
                'next_offset': offset + limit if offset + limit < len(files) else None}

    @mcp.tool(annotations=readonly)
    def evidence_read(scope: str, path: str, offset: int = 0, max_bytes: int = 64000) -> CallToolResult:
        """Read bounded text evidence or a native PNG/JPEG image. Large binary datasets and model weights are never dumped into context."""
        p = inside(roots[scope], path)
        if not p.is_file() or not 0 <= offset or not 1 <= max_bytes <= 256000:
            raise ValueError('Invalid file or read bounds')
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            if p.stat().st_size > 8 * 1024**2:
                raise ValueError('Image exceeds 8 MiB; generate a bounded visual inspection')
            mime = 'image/png' if p.suffix.lower() == '.png' else 'image/jpeg'
            return CallToolResult(content=[TextContent(text=json.dumps({'scope': scope, 'path': path, 'sha256': digest(p)})),
                ImageContent(data=base64.b64encode(p.read_bytes()).decode(), mime_type=mime)])
        if p.suffix.lower() not in ('.json', '.jsonl', '.md', '.txt', '.log', '.csv', '.yaml', '.yml', '.toml', '.py'):
            raise ValueError('Use an artifact path for binary media; evidence_read supports text and images')
        with p.open('rb') as f:
            f.seek(offset); raw = f.read(max_bytes)
        return CallToolResult(content=[TextContent(text=json.dumps({'scope': scope, 'path': path,
            'text': raw.decode('utf-8', errors='replace'), 'next_offset': offset + len(raw), 'bytes_total': p.stat().st_size}, ensure_ascii=False))])

    @mcp.tool(annotations=readonly)
    def video_inspect(scope: str, path: str, timestamps_seconds: list[float]) -> CallToolResult:
        """Return 1-4 native image frames from an existing local video, with source hash and requested timestamps. No download or video generation."""
        p = inside(roots[scope], path)
        if p.suffix.lower() not in ('.mp4', '.mov', '.mkv', '.webm') or not p.is_file():
            raise ValueError('Choose a local video in an evidence root')
        if not 1 <= len(timestamps_seconds) <= 4 or p.stat().st_size > 512 * 1024**2:
            raise ValueError('1-4 timestamps; video size must be <=512 MiB')
        sha = digest(p)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(p)],
                               capture_output=True, text=True, timeout=15, check=True)
        duration = float(json.loads(probe.stdout)['format']['duration'])
        if any(not 0 <= t < duration for t in timestamps_seconds):
            raise ValueError('Timestamp outside video duration')
        content = [TextContent(text=json.dumps({'source_sha256': sha, 'scope': scope, 'path': path,
            'requested_timestamps_seconds': timestamps_seconds, 'duration_seconds': duration,
            'selection': 'ffmpeg accurate seek, first decoded frame at requested timestamp; exact source frame ID not inferred',
            'evidence_level': 'media_inspection_only'}))]
        for t in timestamps_seconds:
            result = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', str(p), '-frames:v', '1',
                '-vf', 'scale=960:540:force_original_aspect_ratio=decrease', '-threads', '1',
                '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1'], capture_output=True, timeout=20, check=True)
            if not result.stdout.startswith(b'\x89PNG'):
                raise ValueError('No video frame decoded')
            content.append(ImageContent(data=base64.b64encode(result.stdout).decode(), mime_type='image/png'))
        if digest(p) != sha:
            raise ValueError('Video changed while reading; refresh evidence')
        return CallToolResult(content=content)

    @mcp.tool(annotations=readonly)
    def workflow_list() -> dict[str, Any]:
        """List executable recipes with exact input schemas and unavailable stages with reasons."""
        return {'workflows': jobs.workflows()}

    @mcp.tool(annotations=writes)
    def workflow_plan(workflow_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Create a one-hour hash-bound execution plan. Does not launch anything. Output is a new isolated job directory."""
        return jobs.plan(workflow_id, parameters)

    @mcp.tool(annotations=writes)
    def workflow_submit(plan_id: str, idempotency_key: str) -> dict[str, Any]:
        """Explicitly execute a prepared plan. Reusing a request key never launches a duplicate. Returns a persistent job ID."""
        return jobs.submit(plan_id, idempotency_key)

    @mcp.tool(annotations=readonly)
    def job_status(job_id: str) -> dict[str, Any]:
        """Read persistent execution state and actual exit status, checking recorded process identity on Linux."""
        return jobs.status(job_id)

    @mcp.tool(annotations=readonly)
    def job_logs(job_id: str, offset: int = 0, max_bytes: int = 16000) -> dict[str, Any]:
        """Read a bounded log page; pass next_offset to continue."""
        return jobs.logs(job_id, offset, max_bytes)

    @mcp.tool(annotations=readonly)
    def job_artifacts(job_id: str) -> dict[str, Any]:
        """List actual outputs; inspect with evidence_read. Completion is not scientific acceptance."""
        return jobs.artifacts(job_id)

    @mcp.tool(annotations=writes)
    def agent_note(key: str, sha256: str, frames: list[int], observation: str, agent_name: str) -> dict[str, Any]:
        """Append a source-bound agent observation separately from human review/acceptance logs."""
        if not observation.strip() or len(observation) > 8000 or not agent_name.strip() or len(agent_name) > 200:
            raise ValueError('Observation and agent name required within bounds')
        with lock, file_lock(jobs.root / 'notes.lock'):
            detail = data().detail(key, sha256)
            if len(frames) > 64 or any(f < 0 or f >= detail['frames'] for f in frames):
                raise ValueError('Frame outside source')
            note = {'id': uuid.uuid4().hex, 'kind': 'agent_observation', 'created_unix': now(),
                    'key': key, 'sha256': sha256, 'frames': frames, 'observation': observation,
                    'agent_name': agent_name, 'human_acceptance': False}
            with (jobs.root / 'agent_notes.jsonl').open('a') as f:
                f.write(json.dumps(note, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
            return note

    @mcp.resource('alphamotion://capabilities')
    def capabilities() -> str:
        return json.dumps(toolchain_status(), ensure_ascii=False)

    return mcp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    create_server(args.config).run(transport='stdio')


if __name__ == '__main__':
    main()
