"""Loads ragbench/config.yaml into a Config object with resolved paths.

    from ragbench.config import load_config
    cfg = load_config()
    cfg.models["chat"], cfg.evidence["max_chars"], cfg.path("index_root")

Secrets are never stored on the Config; `openrouter_key()` reads the .env file on
demand and the value must never be printed, logged or written anywhere.
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"

_CACHE = {}


class Config:
    def __init__(self, data, source):
        self.data = data
        self.source = str(source)
        p = data["paths"]
        self.project_root = Path(p["project_root"])

    # section accessors -----------------------------------------------------
    def __getattr__(self, name):
        data = self.__dict__.get("data")
        if data is not None and name in data:
            return data[name]
        raise AttributeError(name)

    def get(self, dotted, default=None):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, key):
        """Absolute Path for paths.<key> (relative entries resolve against project_root)."""
        raw = self.data["paths"][key]
        pth = Path(raw)
        if not pth.is_absolute():
            pth = self.project_root / pth
        return pth

    def ensure_dir(self, key):
        d = self.path(key)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def hash(self, *sections):
        """Stable short hash of the given top-level sections (all when none given)."""
        keys = sections or tuple(sorted(self.data))
        blob = json.dumps({k: self.data.get(k) for k in keys}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def copy_with(self, overrides):
        """New Config with a nested dict merged over this one (for tests and ablations)."""
        data = copy.deepcopy(self.data)
        _merge(data, overrides)
        return Config(data, self.source + "+overrides")

    def openrouter_key(self):
        """Read OPENROUTER_API_KEY from the environment or the project .env. Never print it."""
        v = os.environ.get("OPENROUTER_API_KEY")
        if v:
            return v.strip()
        envp = self.path("env_file")
        if envp.is_file():
            for line in envp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, val = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    return val.strip().strip('"').strip("'")
        return None

    def __repr__(self):
        return "Config(%s)" % self.source


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def load_config(path=None, fresh=False):
    path = Path(path or os.environ.get("RAGBENCH_CONFIG") or DEFAULT_CONFIG)
    key = str(path.resolve())
    if not fresh and key in _CACHE:
        return _CACHE[key]
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Config(data, path)
    _CACHE[key] = cfg
    return cfg
