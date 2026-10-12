"""Exercise the actual SSH stdio transport and save inspectable acceptance evidence."""
import argparse
import asyncio
import base64
import json
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters


PARAMS = StdioServerParameters(command='ssh', args=['-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', 'arenalabs',
    '/home/arenalabs/motion_audit_web/scripts/mcp_stdio.sh'])


def unpack(result):
    if result.is_error:
        raise RuntimeError(str(result.content))
    return result.structured_content or json.loads(result.content[0].text)


async def main(output, submit_smoke):
    output.mkdir(parents=True, exist_ok=True)
    report = {'transport': 'SSH stdio', 'client_mode': 'legacy initialize/tools/list/tools/call', 'checks': []}
    async with Client(PARAMS, mode='legacy', read_timeout_seconds=180) as client:
        listing = await client.list_tools()
        report['tools'] = [x.name for x in listing.tools]
        report['status'] = unpack(await client.call_tool('toolchain_status', {}))
        print('MCP initialized; tools', len(report['tools']), flush=True)
        catalog = unpack(await client.call_tool('motion_catalog', {'limit': 100}))
        report['catalog_total'] = catalog['total']
        (output/'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
        candidates = [x for x in catalog['items'] if not x.get('error')]
        if not candidates:
            raise RuntimeError('No supported real source items')
        item = next((x for x in candidates if 'kick' in x.get('clip_id', '')), candidates[0])
        key, sha = item['key'], item['sha256']
        detail = unpack(await client.call_tool('motion_inspect', {'key': key, 'expected_sha256': sha}))
        frame = min(100, detail['frames']-1)
        metrics = unpack(await client.call_tool('motion_frame_metrics', {'key': key, 'sha256': sha, 'start': frame}))
        report.update(item_key=key, source_sha256=sha, frame=frame, metrics=metrics)
        result = await client.call_tool('motion_visualize', {'key': key, 'sha256': sha, 'frames': [frame],
            'mode': 'compare', 'view': 'iso', 'show_surface': True, 'show_human': True})
        report['visual'] = unpack(result)
        images = [x for x in result.content if x.type == 'image']
        assert len(images) == 1
        (output/'real-motion-compare.png').write_bytes(base64.b64decode(images[0].data))
        report['checks'].append('real_source_native_image_and_metrics')
        print('Real source visual returned', key, frame, flush=True)
        if submit_smoke:
            plan = unpack(await client.call_tool('workflow_plan', {'workflow_id': 'foundation_smoke', 'parameters': {}}))
            job = unpack(await client.call_tool('workflow_submit', {'plan_id': plan['plan_id'], 'idempotency_key': 'mcp-acceptance-' + plan['plan_id']}))
            report['job_id'] = job['job_id']
            duplicate = unpack(await client.call_tool('workflow_submit', {'plan_id': plan['plan_id'], 'idempotency_key': 'mcp-acceptance-' + plan['plan_id']}))
            assert job['job_id'] == duplicate['job_id']
            report['checks'].append('idempotent_remote_submit')
            (output/'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    # A different MCP process reconnects to persistent state after disconnect.
    if submit_smoke:
        async with Client(PARAMS, mode='legacy', read_timeout_seconds=60) as client:
            for _ in range(60):
                state = unpack(await client.call_tool('job_status', {'job_id': report['job_id']}))
                if state['status'] not in ('submitted', 'running', 'waiting_for_slot'):
                    break
                await asyncio.sleep(2)
            report['job_status'] = state
            report['job_artifacts'] = unpack(await client.call_tool('job_artifacts', {'job_id': report['job_id']}))
            report['job_logs'] = unpack(await client.call_tool('job_logs', {'job_id': report['job_id']}))
            if state['status'] != 'completed':
                (output/'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
                raise RuntimeError('Real foundation smoke did not complete: ' + str(state))
            report['checks'].append('remote_job_survived_client_disconnect_and_completed')
            print('Remote smoke completed', state['exit_code'], flush=True)
    (output/'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Evidence saved', output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--submit-smoke', action='store_true')
    a = p.parse_args(); asyncio.run(main(a.output, a.submit_smoke))
