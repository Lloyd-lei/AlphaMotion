"""HTTP integration; source datasets remain read-only, reviews are explicit POSTs."""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware

from .audit import AuditError, AuditStore, StaleSnapshotError, UI_VERSION, sha256_file
from .wire import pack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(f'{SRCFG.data_root}')


def create_app(*, service=None, audit_root=None, data_root=DATA_ROOT, mode='production',
               idle_seconds=8 * 3600, on_idle=None, gpu_service=None, human_service=None, progress=None):
    if mode not in {'production', 'test'}:
        raise ValueError('mode must be production or test')
    data_root = Path(data_root).resolve()
    if service is None:
        from .adapters import FactoryService
        service = FactoryService(data_root, PROJECT_ROOT / 'cache' / 'arrays')
    from .atlas_routes import atlas_services
    gpu_service, human_service, progress = atlas_services(data_root, PROJECT_ROOT / 'cache' / 'atlas',
                                                          gpu_service, human_service, progress)
    audit_root = Path(os.path.abspath(audit_root or data_root / 'audit'))
    if mode == 'test' and not audit_root.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError('test audit directory must be inside this project')
    audit = AuditStore(audit_root, allowed_data_root=data_root)
    last_activity = time.monotonic()
    # Shared MuJoCo state and cache creation are serialized; the event loop and
    # static UI remain responsive while bounded CPU work runs in one worker.
    operation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        async def watchdog():
            while True:
                await asyncio.sleep(min(30, max(1, idle_seconds / 2)))
                if time.monotonic() - app.state.last_activity >= idle_seconds:
                    if on_idle:
                        on_idle()
                    return
        task = asyncio.create_task(watchdog())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title='动作人工审计', version=UI_VERSION, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=1)
    app.state.last_activity = last_activity
    app.state.audit = audit
    app.state.service = service

    @app.middleware('http')
    async def local_request(request: Request, call_next):
        host = request.url.hostname
        if host not in {'127.0.0.1', 'localhost', '::1', 'testserver'}:
            return JSONResponse({'detail': {'code': 'host_not_allowed', 'message': '仅接受本机访问'}}, 403)
        app.state.last_activity = time.monotonic()
        if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            origin = request.headers.get('origin')
            if origin and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': {'code': 'origin_not_allowed', 'message': '仅接受同源审计提交'}}, 403)
            if request.headers.get('x-audit-request') != '1':
                return JSONResponse({'detail': {'code': 'request_header_required', 'message': '缺少审计请求标识'}}, 403)
            if request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return JSONResponse({'detail': {'code': 'json_required', 'message': '提交必须是 JSON'}}, 415)
            try:
                length = int(request.headers.get('content-length', '0') or 0)
            except ValueError:
                return JSONResponse({'detail': {'code': 'invalid_length', 'message': '请求长度无效'}}, 400)
            if length < 0:
                return JSONResponse({'detail': {'code': 'invalid_length', 'message': '请求长度无效'}}, 400)
            if length > 1_048_576:
                return JSONResponse({'detail': {'code': 'request_too_large', 'message': '提交内容过长'}}, 413)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(AuditError)
    async def audit_error(request, exc):
        return JSONResponse({'detail': exc.as_dict()}, 409 if isinstance(exc, StaleSnapshotError) else 422)

    async def compute(fn, *args, **kwargs):
        async with operation_lock:
            try:
                return await run_in_threadpool(fn, *args, **kwargs)
            except AuditError:
                raise
            except KeyError as exc:
                raise HTTPException(404, {'code': 'item_not_found', 'message': str(exc)}) from exc
            except (ValueError, FileNotFoundError) as exc:
                # Adapter exceptions are mapped by their explicit code when
                # supplied, never rendered as guessed or fabricated geometry.
                code = getattr(exc, 'code', 'data_error')
                stale = 'stale' in code or 'snapshot' in code
                raise HTTPException(409 if stale else 422, {'code': code, 'message': str(exc), 'reload_required': stale}) from exc

    def current_catalog():
        result = service.catalog()
        for item in result['items']:
            if item.get('error') or item.get('supported') is False:
                # A queue hash is historical evidence, never proof of a current
                # readable file when adapter inspection has failed.
                item.setdefault('queue_npz_sha256', item.get('npz_sha256'))
                item['npz_sha256'] = None
                item['sha256'] = None
                item['current_hash_known'] = False
        return result

    def catalog_with_reviews():
        result = current_catalog()
        stats = audit.analytics(result['items'])
        reviews = stats.get('by_item')
        if reviews is None:
            reviews = audit.item_statuses(result['items'])
        by_item = reviews if isinstance(reviews, dict) else {r['item_id']: r for r in reviews}
        for item in result['items']:
            item['review'] = by_item.get(item['item_id'], {})
        return {**result, 'mode': mode, 'ui_version': UI_VERSION}

    @app.get('/api/health')
    async def health():
        return {'status': 'ok', 'mode': mode, 'ui_version': UI_VERSION,
                'idle_timeout_hours': idle_seconds / 3600, 'source_read_only': True}

    @app.get('/api/catalog')
    async def catalog():
        return await compute(catalog_with_reviews)

    @app.get('/api/items/{key}')
    async def detail(key: str):
        return await compute(service.detail, key)

    @app.get('/api/items/{key}/scene')
    async def scene(key: str, sha256: str = Query(min_length=64, max_length=64)):
        def build():
            meta, arrays = service.scene(key, sha256)
            return pack(meta, arrays)
        return Response(await compute(build), media_type='application/octet-stream')

    @app.get('/api/items/{key}/chunk')
    async def chunk(key: str, sha256: str = Query(min_length=64, max_length=64),
                    start: int = Query(0, ge=0), count: int = Query(16, ge=1, le=64),
                    mode: str = Query('corrected', pattern='^(corrected|raw)$')):
        def build():
            meta, arrays = service.chunk(key, start, count, mode, sha256)
            return pack(meta, arrays)
        return Response(await compute(build), media_type='application/octet-stream')

    @app.get('/api/audit/status')
    async def status():
        return await compute(lambda: {**audit.analytics(current_catalog()['items']), 'mode': mode})

    async def body(request):
        raw = bytearray()
        async for part in request.stream():
            if len(raw) + len(part) > 1_048_576:
                raise HTTPException(413, '提交内容过长')
            raw.extend(part)
        try:
            from .audit import decode_line
            value = decode_line(raw)
        except ValueError as exc:
            raise HTTPException(422, {'code': 'invalid_json', 'message': str(exc)}) from exc
        if not isinstance(value, dict):
            raise HTTPException(422, '提交必须是 JSON 对象')
        value.setdefault('ts', datetime.now(timezone.utc).isoformat())
        return value

    @app.post('/api/decisions')
    async def decision(request: Request):
        value = await body(request)
        key = value.pop('key', None)
        if not isinstance(key, str) or not key:
            raise HTTPException(422, '缺少当前条目标识 key')
        value.setdefault('ui_version', UI_VERSION)
        def append():
            viewed_hash = value.get('npz_sha256')
            item = service.detail(key, expected_sha=viewed_hash)
            if item['item_id'] != value.get('item_id'):
                raise HTTPException(422, 'item_id 与当前预览不一致')
            source_path = Path(item['npz'])
            metadata = {'item_id': item['item_id'], 'npz_sha256': viewed_hash, 'frames': item['frames']}
            return audit.append_decision(value, metadata,
                lambda: sha256_file(source_path, allowed_root=data_root))
        record = await compute(append)
        return {'status': 'saved', 'record': record, 'mode': mode}

    @app.post('/api/issues')
    async def issue(request: Request):
        value = await body(request)
        return {'status': 'saved', 'record': await compute(audit.append_issue, value), 'mode': mode}

    from .atlas_routes import register_atlas_routes
    register_atlas_routes(app, compute, gpu_service, human_service, progress)

    dist = PROJECT_ROOT / 'frontend' / 'dist'

    @app.get('/{path:path}', include_in_schema=False)
    async def static(path: str):
        candidate = (dist / path).resolve()
        if not candidate.is_relative_to(dist.resolve()):
            raise HTTPException(404)
        if candidate.is_file():
            return FileResponse(candidate, headers={'Cache-Control': 'no-store'} if candidate.name == 'index.html' else None)
        if path.startswith('api/'):
            raise HTTPException(404)
        if not (dist / 'index.html').exists():
            return JSONResponse({'detail': '前端尚未构建，请先运行 npm run build'}, 503)
        return FileResponse(dist / 'index.html', headers={'Cache-Control': 'no-store'})

    return app
