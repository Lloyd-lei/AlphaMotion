"""Single place for machine-specific paths.

Reads configs/sr.example.toml, then configs/sr.toml (optional, gitignored), then env vars SR_<KEY>.
Values are plain strings (so f'{CFG.data_root}/raw' == the old literal); `CFG.path(key)` gives a Path.
Stdlib only and Python >= 3.10 (the genmo-gem env is 3.10, so tomllib is optional).

CLI:
  python surface_retargeting/config.py            # print resolved values
  python surface_retargeting/config.py --export   # 'export SR_...=...' lines for shell scripts
  python surface_retargeting/config.py --get KEY
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / 'configs' / 'sr.example.toml'
LOCAL = REPO / 'configs' / 'sr.toml'
PRIMARY = ('data_root', 'work', 'umr_root', 'body_models', 'robot_assets_root', 'python_env')

# Literal prefixes that config JSONs (registry, umr_configs, ...) store as ${SR_<KEY>} tokens, longest first.
TOKEN_KEYS = ('gmr_root', 'research_root', 'umr_root', 'python_env', 'work', 'hiphi2smplx_root', 'h2s_shim',
              'beta_fit', 'factory_root', 'robot_assets_root', 'am2exp_root', 'data_root', 'dataset_root',
              'viewer_home')


def _parse_toml(text: str) -> dict:
    try:
        import tomllib  # py >= 3.11
        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    out = {}  # minimal subset: [section] / key = "string" / comments
    cur = out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'^\[([\w.]+)\]$', line)
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        m = re.match(r'^(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(#.*)?$', line)
        if not m:
            raise ValueError(f'unsupported TOML line (only key = "string"): {raw!r}')
        cur[m.group(1)] = json.loads('"' + m.group(2) + '"')
    return out


class Config:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        raw = dict(_parse_toml(EXAMPLE.read_text()).get('paths', {}))
        if LOCAL.exists():
            raw.update(_parse_toml(LOCAL.read_text()).get('paths', {}))
        for k in list(raw):
            v = env.get('SR_' + k.upper())
            if v:
                raw[k] = v
        self._raw = raw
        self._vals = {'repo': str(REPO)}
        for k in raw:
            self._resolve(k, ())

    def _resolve(self, k, stack):
        if k in self._vals:
            return self._vals[k]
        if k in stack:
            raise ValueError(f'config cycle: {stack + (k,)}')
        v = re.sub(r'\{(\w+)\}', lambda m: self._resolve(m.group(1), stack + (k,)), self._raw[k])
        self._vals[k] = v
        return v

    def __getattr__(self, k):
        try:
            return self.__dict__['_vals'][k]
        except KeyError:
            raise AttributeError(f'unknown config key {k!r}') from None

    def path(self, k) -> Path:
        return Path(self._vals[k])

    def as_dict(self) -> dict:
        return dict(self._vals)

    # ---- derived helpers ----
    @property
    def umr_python(self) -> str:
        return f'{self.python_env}/bin/python'

    @property
    def umr_tools(self) -> str:
        """UMR wrapper scripts (umr_pipe.py / umr_run.py), now part of this repo."""
        return str(REPO / 'surface_retargeting' / 'retarget' / 'umr_wrappers')

    @property
    def registry_json(self) -> str:
        return str(REPO / 'configs' / 'robots' / 'registry.json')

    @property
    def draft_configs(self) -> str:
        return str(REPO / 'configs' / 'umr_configs')

    def add_umr_to_path(self):
        """UMR upstream is a script tree, not a package: put UMR/scripts and UMR on sys.path."""
        for p in (f'{self.umr_root}/scripts', self.umr_root):
            if p not in sys.path:
                sys.path.insert(0, p)

    def add_hiphi2smplx_to_path(self, shim=True):
        """hiphi2smplx is used from source (pinned submodule); genmo-gem has no h5py, hence the shim."""
        ps = [f'{self.hiphi2smplx_root}/src'] + ([self.h2s_shim] if shim else [])
        for p in reversed(ps):
            if p not in sys.path:
                sys.path.insert(0, p)

    # ---- ${SR_*} tokens in config JSON ----
    def expand(self, obj):
        """Replace ${SR_KEY} tokens in every string of a JSON-like object."""
        if isinstance(obj, str):
            return re.sub(r'\$\{SR_(\w+)\}', lambda m: self._vals[m.group(1).lower()], obj)
        if isinstance(obj, list):
            return [self.expand(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.expand(v) for k, v in obj.items()}
        return obj

    def tokenize(self, obj):
        """Inverse of expand for the machine-specific prefixes in TOKEN_KEYS."""
        if isinstance(obj, str):
            for k in TOKEN_KEYS:
                v = self._vals[k]
                if obj == v or obj.startswith(v + '/'):
                    return '${SR_' + k.upper() + '}' + obj[len(v):]
            return obj
        if isinstance(obj, list):
            return [self.tokenize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.tokenize(v) for k, v in obj.items()}
        return obj

    def load_json(self, path):
        with open(path) as f:
            return self.expand(json.load(f))


CFG = Config()


def _main(argv):
    if argv[:1] == ['--export']:
        import shlex
        for k, v in CFG.as_dict().items():
            print(f'export SR_{k.upper()}={shlex.quote(v)}')
        print(f'export SR_UMR_PYTHON={shlex.quote(CFG.umr_python)}')
        print(f'export SR_UMR_TOOLS={shlex.quote(CFG.umr_tools)}')
    elif argv[:1] == ['--get']:
        print(getattr(CFG, argv[1]))
    else:
        for k, v in CFG.as_dict().items():
            print(f'{k:18s} {v}{"   (primary)" if k in PRIMARY else ""}')


if __name__ == '__main__':
    _main(sys.argv[1:])
