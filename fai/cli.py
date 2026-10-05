"""Dataclass config'lerini argparse'a bağlar: her alan --alan-adi olur (PipelineConfig, PostConfig).
Böylece eşik eklendiğinde CLI ile config ayrışmaz."""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
from typing import Iterable


def _opt_float(s: str):
    return None if str(s).lower() in ("none", "yok", "off") else float(s)


def add_dataclass_args(ap: argparse.ArgumentParser, cls, skip: Iterable[str] = (), title: str = "") -> None:
    grp = ap.add_argument_group(title or cls.__name__)
    for f in fields(cls):
        if f.name in skip:
            continue
        default = f.default if f.default is not MISSING else None
        flag = "--" + f.name.replace("_", "-")
        optional = default is None or str(f.type).startswith("Optional")
        if isinstance(default, tuple):
            grp.add_argument(flag, type=float, nargs="+", default=list(default), help=f"varsayılan: {' '.join(map(str, default))}")
        elif optional:
            grp.add_argument(flag, type=_opt_float, default=default, help=f"varsayılan: {default}; 'none' = kapalı")
        elif isinstance(default, bool):
            grp.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes", "evet"), default=default)
        elif isinstance(default, int):
            grp.add_argument(flag, type=int, default=default, help=f"varsayılan: {default}")
        else:
            grp.add_argument(flag, type=float, default=default, help=f"varsayılan: {default}")


def dataclass_from_args(cls, ns: argparse.Namespace, **overrides):
    kw = {}
    for f in fields(cls):
        if hasattr(ns, f.name):
            v = getattr(ns, f.name)
            kw[f.name] = tuple(v) if isinstance(f.default, tuple) else v
    kw.update(overrides)
    return cls(**kw)
