"""HTTP contracts use isolated fixtures, never the real human review logs."""
import hashlib
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app import create_app, PROJECT_ROOT
from backend.audit import StaleSnapshotError
from backend.wire import unpack

HEADERS = {'X-Audit-Request': '1'}


class FixtureService:
    def __init__(self, root):
        self.path = root / 'motion.npz'
        self.path.write_bytes(b'isolated-adapter-contract-fixture')

    def detail(self, key, expected_sha=None):
        if key != 'fixture-key':
            raise KeyError(key)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if expected_sha and digest != expected_sha:
            raise StaleSnapshotError('fixture changed')
        return {'key': key, 'item_id': 'fixture-item', 'npz': str(self.path),
                'npz_sha256': digest, 'frames': 10, 'fps': 30,
                'source': 'fixture', 'robot': 'fixture', 'queued': False}

    def catalog(self):
        return {'items': [self.detail('fixture-key')], 'warnings': []}

    def scene(self, key, sha):
        self.detail(key, sha)
        return {'npz_sha256': sha}, {'points': np.array([[0, 1, .0005]], dtype=np.float32)}

    def chunk(self, key, start, count, mode, sha):
        self.detail(key, sha)
        if start >= 10:
            raise ValueError('frame outside range')
        return {'start': start, 'mode': mode, 'count': min(count, 10-start)}, {'flags': np.zeros((min(count, 10-start), 13), np.uint8)}


@pytest.fixture
def context():
    (PROJECT_ROOT / 'qa').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='http-fixture-', dir=PROJECT_ROOT / 'qa') as name:
        root = Path(name)
        (root / 'audit').mkdir()
        service = FixtureService(root)
        app = create_app(service=service, data_root=root, audit_root=root/'audit', mode='test')
        with TestClient(app) as client:
            yield client, service, root


def payload(service):
    item = service.detail('fixture-key')
    return {'key': item['key'], 'item_id': item['item_id'], 'npz_sha256': item['npz_sha256'],
            'verdict': 'needs_fix', 'frame_ranges': [[2, 5]], 'reasons': ['hover'],
            'note': 'isolated integration test', 'reviewer': 'fixture-reviewer'}


def test_read_contract_and_binary(context):
    client, service, root = context
    assert client.get('/api/health').json()['mode'] == 'test'
    catalog = client.get('/api/catalog').json()
    assert catalog['items'][0]['review']['status'] == 'pending'
    item = client.get('/api/items/fixture-key').json()
    result = client.get('/api/items/fixture-key/scene', params={'sha256': item['npz_sha256']})
    assert result.status_code == 200
    assert result.headers['content-type'] == 'application/octet-stream'
    meta, arrays = unpack(result.content)
    assert arrays['points'].dtype == np.dtype('float32')
    assert abs(float(arrays['points'][0, 2]) - .0005) < 1e-7
    result = client.get('/api/items/fixture-key/chunk', params={'sha256': item['npz_sha256'], 'start': 8, 'count': 4, 'mode': 'raw'})
    assert unpack(result.content)[1]['flags'].shape == (2, 13)
    assert not list((root/'audit').iterdir())


def test_actual_append_http_contract_and_stale(context):
    client, service, root = context
    body = payload(service)
    response = client.post('/api/decisions', json=body, headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()['record']['frame_ranges'] == [[2, 5]]
    assert response.json()['record']['ui_version']
    assert response.json()['record']['ts'].endswith('+00:00')
    logfile = root/'audit/decisions.jsonl'
    original = logfile.read_bytes()
    service.path.write_bytes(b'changed-fixture')
    response = client.post('/api/decisions', json=body, headers=HEADERS)
    assert response.status_code == 409
    assert response.json()['detail']['reload_required'] is True
    assert logfile.read_bytes() == original
    assert client.get('/api/audit/status').json()['stale'] == 1


def test_write_origins_bounds_and_validation(context):
    client, service, root = context
    body = payload(service)
    assert client.post('/api/decisions', json=body).status_code == 403
    assert client.post('/api/decisions', json=body, headers={**HEADERS, 'Origin': 'https://unrelated.example'}).status_code == 403
    bad = dict(body, item_id='another-item')
    assert client.post('/api/decisions', json=bad, headers=HEADERS).status_code == 422
    bad = dict(body, frame_ranges=[[0, 11]])
    assert client.post('/api/decisions', json=bad, headers=HEADERS).status_code == 422
    assert client.get('/api/items/missing').status_code == 404
    assert client.get('/api/items/fixture-key/scene').status_code == 422
    assert client.get('/api/items/fixture-key/chunk', params={'sha256': body['npz_sha256'], 'count': 65}).status_code == 422
    assert client.get('/api/items/fixture-key/chunk', params={'sha256': body['npz_sha256'], 'mode': 'guessed'}).status_code == 422
    assert client.post('/api/decisions', content='{"key":1,"key":2}', headers={**HEADERS,'Content-Type':'application/json'}).status_code == 422
    assert not list((root/'audit').iterdir())


def test_issue_write_and_no_implicit_random_cohort(context):
    client, service, root = context
    issue = {'scope': 'robot', 'target': 'fixture', 'reasons': ['jitter'], 'note': 'test only',
             'example_items': ['fixture-item'], 'reviewer': 'fixture-reviewer'}
    assert client.post('/api/issues', json=issue, headers=HEADERS).status_code == 200
    assert (root/'audit/issues.jsonl').exists()
    assert not client.get('/api/audit/status').json()['random_sample']['available']


def test_streaming_body_limit(context):
    client, service, root = context
    def oversized():
        yield b'{"note":"'
        for _ in range(18):
            yield b'x' * 65536
        yield b'"}'
    result = client.post('/api/issues', content=oversized(),
        headers={**HEADERS, 'Content-Type': 'application/json'})
    assert result.status_code == 413
    assert not list((root/'audit').iterdir())


def test_http_preserves_audit_symlink_rejection(context):
    client, service, root = context
    linked = root/'linked-audit'
    linked.symlink_to(root/'audit', target_is_directory=True)
    with pytest.raises(OSError):
        create_app(service=service, data_root=root, audit_root=linked, mode='test')
    assert not list((root/'audit').iterdir())


def test_unreadable_item_does_not_reuse_queue_hash(context):
    client, service, root = context
    body = payload(service)
    body.update(verdict='accept', frame_ranges=[])
    assert client.post('/api/decisions', json=body, headers=HEADERS).status_code == 200
    stale_row = service.detail('fixture-key')
    stale_row.update(error='source file missing', supported=False)
    service.catalog = lambda: {'items': [dict(stale_row)], 'warnings': []}
    result = client.get('/api/catalog').json()['items'][0]
    assert result['npz_sha256'] is None
    assert result['review']['status'] == 'stale'
    assert client.get('/api/audit/status').json()['accepted'] == 0


def test_idle_lifetime_is_request_based(context):
    _, service, root = context
    expired = []
    app = create_app(service=service, data_root=root, audit_root=root/'audit', mode='test',
                     idle_seconds=.1, on_idle=lambda: expired.append(True))
    with TestClient(app) as client:
        client.get('/api/health')
        time.sleep(1.2)
        assert expired == [True]
