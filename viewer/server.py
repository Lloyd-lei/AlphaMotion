"""Single local service process, shut down after eight hours without requests."""
import argparse
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import uvicorn
from backend.app import create_app, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('production', 'test'), default='production')
    args = parser.parse_args()
    audit_root = None
    if args.mode == 'test':
        audit_root = PROJECT_ROOT / 'qa' / 'fixture_audit'
        audit_root.mkdir(parents=True, exist_ok=True)
    server = None
    app = create_app(mode=args.mode, audit_root=audit_root,
                     on_idle=lambda: setattr(server, 'should_exit', True))
    config = uvicorn.Config(app, host='127.0.0.1', port=8765, workers=1,
                            access_log=False, limit_concurrency=32,
                            timeout_keep_alive=5, timeout_graceful_shutdown=30)
    server = uvicorn.Server(config)
    server.run()


if __name__ == '__main__':
    main()
