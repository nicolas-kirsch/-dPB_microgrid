"""Plain-text and JSON records of one training run, written next to its checkpoints.

The checkpoints remain the authoritative state; these files only make a run
inspectable without torch (e.g. after a batch job): ``train.log`` mirrors the
console, ``run.json`` the command line and configuration, ``history.json`` the
completed epochs and ``summary.json`` the final report.
"""
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return str(value)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_jsonable))
    temporary.replace(path)


def _git(root, *command):
    try:
        return subprocess.run(["git", *command], cwd=root, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parameter_count(model):
    """Real parameters, counting complex entries twice."""
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())


def start_run_log(output, *, args, cfg, model, architecture, root):
    """Mirror stdout/stderr into output/train.log and write output/run.json."""
    log = open(output / "train.log", "a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
    status = _git(root, "status", "--porcelain")
    write_json(output / "run.json", {
        "architecture": architecture,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "args": vars(args),
        "init_seed": cfg.get("init_seed"), "data_seed": cfg["seed"],
        "parameter_count": parameter_count(model),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "config": cfg})
    print(f"Run directory: {output.resolve()}", flush=True)
