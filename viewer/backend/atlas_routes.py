"""HTTP routes for the atlas pages: progress, gpu_npz_v2 viewer, raw SMPL-X browser.

All handlers are GET-only and read-only; heavy work goes through the app's
serialized ``compute`` (one CPU worker, bounded chunks <= 64 frames).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Query
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .wire import pack


def finite(value):
    """JSON responses refuse NaN/Inf; report them as null instead of failing."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float('inf') else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def atlas_services(data_root, cache_root, gpu_service=None, human_service=None, progress=None):
    cache_root = Path(cache_root)
    if gpu_service is None:
        from .gpu_adapter import GpuService
        gpu_service = GpuService(data_root, cache_root / 'arrays')
    if human_service is None:
        from .human_smplx import HumanService
        cache_root.mkdir(parents=True, exist_ok=True)
        human_service = HumanService(data_root, cache_root)
    if getattr(gpu_service, 'source_resolver', 'absent') is None:
        gpu_service.source_resolver = lambda name, path=None: (human_service.index.ensure(), human_service.index.find_by_clip_name(name, path))[1]
    if getattr(gpu_service, 'smplx_model', 'absent') is None:
        gpu_service.smplx_model = lambda: human_service.model
    if progress is None:
        from .progress import ProgressScanner
        progress = ProgressScanner(data_root, cache_root, human_service.index)
    return gpu_service, human_service, progress


def register_atlas_routes(app, compute, gpu, human, progress):
    app.state.gpu_service = gpu
    app.state.human_service = human
    app.state.progress = progress

    @app.get('/api/atlas/progress')
    async def atlas_progress(force: bool = False):
        # Never blocks on a scan: returns the last result and starts a background
        # rescan only when the previous one is older than 5 minutes.
        return finite(await run_in_threadpool(progress.get, force))

    @app.get('/api/gpu/catalog')
    async def gpu_catalog(q: str = '', source: str = '', pipeline: str = '',
                          offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)):
        return finite(await compute(gpu.catalog, q, source, pipeline, offset, limit))

    @app.get('/api/gpu/items/{key}')
    async def gpu_detail(key: str):
        return finite(await compute(gpu.detail, key))

    @app.get('/api/gpu/items/{key}/scene')
    async def gpu_scene(key: str, sha256: str = Query(min_length=64, max_length=64)):
        def build():
            meta, arrays = gpu.scene(key, sha256)
            return pack(finite(meta), arrays)
        return Response(await compute(build), media_type='application/octet-stream')

    @app.get('/api/gpu/items/{key}/chunk')
    async def gpu_chunk(key: str, sha256: str = Query(min_length=64, max_length=64),
                        start: int = Query(0, ge=0), count: int = Query(32, ge=1, le=64), human: bool = True):
        def build():
            meta, arrays = gpu.chunk(key, start, count, sha256, human)
            return pack(finite(meta), arrays)
        return Response(await compute(build), media_type='application/octet-stream')

    @app.get('/api/human/catalog')
    async def human_catalog(source: str = '', variant: str = '', category: str = '', q: str = '',
                            sort: str = Query('source', pattern='^(source|duration|residual)$'),
                            offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)):
        return finite(await compute(human.catalog, source, variant, category, q, sort, offset, limit))

    @app.get('/api/human/items/{clip_id}')
    async def human_detail(clip_id: str):
        return finite(await compute(human.detail, clip_id))

    @app.get('/api/human/items/{clip_id}/scene')
    async def human_scene(clip_id: str):
        def build():
            meta, arrays = human.scene(clip_id)
            return pack(finite(meta), arrays)
        return Response(await compute(build), media_type='application/octet-stream')

    @app.get('/api/human/items/{clip_id}/chunk')
    async def human_chunk(clip_id: str, start: int = Query(0, ge=0), count: int = Query(32, ge=1, le=64)):
        def build():
            meta, arrays = human.chunk(clip_id, start, count)
            return pack(finite(meta), arrays)
        return Response(await compute(build), media_type='application/octet-stream')
