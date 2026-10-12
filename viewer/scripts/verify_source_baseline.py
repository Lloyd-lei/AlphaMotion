"""Read-only comparison; writes evidence only inside the audit web project."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    baseline = json.loads((ROOT/'qa/source_baseline.json').read_text())
    rows = []
    for name, old in baseline['npz_baseline'].items():
        p = Path(name)
        stat = p.stat()
        current = {'sha256': digest(p), 'size': stat.st_size,
                   'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}
        rows.append({'path': name, 'unchanged': old == current, 'current': current})
    documents = [{'path': name, 'unchanged': digest(Path(name)) == old}
                 for name, old in baseline['source_documents'].items()]
    logs = []
    for name in ('decisions.jsonl', 'issues.jsonl'):
        path = Path(f'{SRCFG.data_root}/audit')/name
        logs.append({'path': str(path), 'exists': path.exists(),
                     'bytes': path.stat().st_size if path.exists() else 0,
                     'sha256': digest(path) if path.exists() else None})
    result = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'all_npz_unchanged': all(x['unchanged'] for x in rows),
              'all_documents_unchanged': all(x['unchanged'] for x in documents),
              'npz': rows, 'documents': documents, 'formal_audit_logs': logs}
    (ROOT/'qa/source-preservation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('npz','documents')},ensure_ascii=False))


if __name__ == '__main__':
    main()
