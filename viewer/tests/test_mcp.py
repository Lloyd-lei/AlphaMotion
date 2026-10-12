"""MCP protocol tests: native images, stale sources, persistence and idempotency."""
import asyncio
import base64
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

from mcp import Client
import numpy as np
from PIL import Image
import pytest

from agent_tools.server import create_server
from agent_tools.jobs import Jobs
from agent_tools.common import file_lock


class Fixture:
    sha = 'a' * 64

    def detail(self, key, expected_sha=None):
        if key != 'fixture' or (expected_sha and expected_sha != self.sha):
            raise ValueError('Stale or missing fixture')
        return {'key': key, 'sha256': self.sha, 'frames': 10, 'fps': 30, 'item_id': 'synthetic_test_only'}

    def catalog(self):
        return {'items': [self.detail('fixture')], 'warnings': ['Synthetic protocol test, not source-data acceptance']}

    def scene(self, key, sha):
        d = self.detail(key, sha)
        return {**d, 'geoms': [{'id': 0, 'type': 'mesh', 'mesh': 'tri', 'rgba': [.3, .4, .5, 1]}]}, {
            'tri_vertices': np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]]),
            'tri_faces': np.array([[0, 1, 2]]), 'surface_geom': np.zeros(3, dtype=int),
            'surface_points': np.array([[0, 0, -.001], [1, 0, 1], [0, 1, 0]])}

    def chunk(self, key, start, count, mode, sha):
        d = self.detail(key, sha)
        if not 0 <= start < 10 or not 1 <= count <= 64:
            raise ValueError('Invalid frame')
        count = min(count, 10-start)
        return {**d, 'mode': mode, 'coordinate_system': 'Z-up metres', 'warnings': [], 'labels_pose': 'corrected'}, {
            'geom_pos': np.zeros((count, 1, 3)), 'geom_rot': np.tile(np.eye(3), (count, 1, 1, 1)),
            'frame_indices': np.arange(start, start+count), 'ground_point_contact': np.tile(np.array([128], dtype=np.uint8), (count, 1))}


@pytest.fixture
def configured(tmp_path):
    script = tmp_path / 'action.py'
    script.write_text('import pathlib,sys,time\np=pathlib.Path(sys.argv[1]);time.sleep(.15);(p/"done.json").write_text(\'{"fixture":true}\');print("job complete")\n')
    config = {'data_root': str(tmp_path), 'cache_root': str(tmp_path/'cache'), 'state_root': str(tmp_path/'state'),
              'reserve_bytes': 0, 'evidence_roots': {'fixture': str(tmp_path)}, 'workflows': {
                  'fixture': {'enabled': True, 'stage': 'test', 'description': 'Isolated integration fixture',
                    'argv': [sys.executable, str(script), '{output}'], 'cwd': str(tmp_path),
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                    'code_files': [str(script)], 'timeout_seconds': 10, 'device': 'cpu'}}}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config))
    return path, config, script


def test_protocol_native_visual_and_stale(configured):
    path, _, _ = configured
    async def check():
        async with Client(create_server(path, Fixture())) as client:
            listing = await client.list_tools()
            names = {x.name for x in listing.tools}
            assert {'motion_catalog', 'motion_visualize', 'workflow_submit', 'job_status', 'video_inspect'} <= names
            result = await client.call_tool('motion_visualize', {'key': 'fixture', 'sha256': Fixture.sha,
                'frames': [0], 'mode': 'compare', 'show_human': False})
            assert not result.is_error, result
            assert [x.type for x in result.content] == ['text', 'image']
            report = json.loads(result.content[0].text)
            assert [r['mode'] for r in report['panels']] == ['raw', 'corrected']
            assert report['panels'][1]['points_below_ground_0_5mm'] == [1]
            assert 'ground_source_contact_points' not in report['panels'][0]
            Image.open(io.BytesIO(base64.b64decode(result.content[1].data))).verify()
            stale = await client.call_tool('motion_visualize', {'key': 'fixture', 'sha256': 'b'*64, 'frames': [0]})
            assert stale.is_error
            outside = await client.call_tool('evidence_read', {'scope': 'fixture', 'path': '../outside.json'})
            assert outside.is_error
    asyncio.run(check())


def test_persistent_job_and_idempotency(configured):
    path, cfg, _ = configured
    jobs = Jobs(cfg, path)
    plan = jobs.plan('fixture', {})
    first = jobs.submit(plan['plan_id'], 'repeat-request')
    second = jobs.submit(plan['plan_id'], 'repeat-request')
    assert first['job_id'] == second['job_id']
    for _ in range(200):
        state = Jobs(cfg, path).status(first['job_id'])
        if state['status'] in ('completed', 'failed'):
            break
        time.sleep(.05)
    assert state['status'] == 'completed', jobs.logs(first['job_id'])
    assert state['exit_code'] == 0
    assert len(jobs.artifacts(first['job_id'])['artifacts']) == 1
    assert 'job complete' in jobs.logs(first['job_id'])['text']
    other = jobs.plan('fixture', {})
    with pytest.raises(ValueError, match='different plan'):
        jobs.submit(other['plan_id'], 'repeat-request')


def test_changed_plan_source_and_schema_rejected(configured):
    path, cfg, script = configured
    jobs = Jobs(cfg, path)
    with pytest.raises(Exception):
        jobs.plan('fixture', {'arbitrary_command': 'x'})
    plan = jobs.plan('fixture', {})
    script.write_text('raise RuntimeError("changed")')
    with pytest.raises(ValueError, match='changed'):
        jobs.submit(plan['plan_id'], 'request')
    assert not (jobs.root/'jobs'/plan['plan_id']).exists()


def test_agent_notes_do_not_become_human_decisions(configured):
    path, cfg, _ = configured
    async def check():
        async with Client(create_server(path, Fixture())) as client:
            result = await client.call_tool('agent_note', {'key': 'fixture', 'sha256': Fixture.sha,
                'frames': [0], 'observation': 'Synthetic observation', 'agent_name': 'test-agent'})
            assert not result.is_error
            assert result.structured_content['human_acceptance'] is False
    asyncio.run(check())
    assert not list(Path(cfg['data_root']).rglob('decisions.jsonl'))


def test_busy_slot_never_runs_duplicate(configured):
    path, cfg, _ = configured
    jobs = Jobs(cfg, path)
    plan = jobs.plan('fixture', {})
    with file_lock(jobs.root/'execution.lock'):
        jobs.submit(plan['plan_id'], 'busy')
        for _ in range(100):
            state = jobs.status(plan['plan_id'])
            if state['status'] == 'failed':
                break
            time.sleep(.05)
        assert state['status'] == 'failed'
        assert not jobs.artifacts(plan['plan_id'])['artifacts']


def test_timeout_and_exit_are_recorded(configured):
    path, cfg, script = configured
    script.write_text('import time\ntime.sleep(5)\n')
    cfg['workflows']['fixture']['timeout_seconds'] = .1
    path.write_text(json.dumps(cfg))
    jobs = Jobs(cfg, path)
    plan = jobs.plan('fixture', {})
    jobs.submit(plan['plan_id'], 'timeout')
    for _ in range(100):
        state = jobs.status(plan['plan_id'])
        if state['status'] in ('timed_out', 'failed'):
            break
        time.sleep(.05)
    assert state['status'] == 'timed_out'
    assert state['exit_code'] != 0


def test_real_video_decoding_to_mcp_images(configured):
    import shutil
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    path, cfg, _ = configured
    video = path.parent/'fixture.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=4',
                    '-t', '1', '-pix_fmt', 'yuv420p', str(video)], check=True)
    async def check():
        async with Client(create_server(path, Fixture())) as client:
            result = await client.call_tool('video_inspect', {'scope': 'fixture', 'path': video.name,
                                                             'timestamps_seconds': [0, .5]})
            assert not result.is_error, result
            assert [x.type for x in result.content] == ['text', 'image', 'image']
            for image in result.content[1:]:
                Image.open(io.BytesIO(base64.b64decode(image.data))).verify()
    asyncio.run(check())
