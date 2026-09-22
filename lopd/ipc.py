from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

IPC_ROOT = Path("/dev/shm")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def ipc_dir_for_step(run_name: str, step: int) -> Path:
    return IPC_ROOT / f"lopd_{_SAFE.sub('_', run_name.strip()) or 'run'}_step_{step}"


def write_inputs(ipc_dir: Path, rollouts: list[dict], config: dict[str, Any]) -> None:
    if ipc_dir.exists():
        shutil.rmtree(ipc_dir)
    (ipc_dir / "rollouts").mkdir(parents=True)
    for i, record in enumerate(rollouts):
        (ipc_dir / "rollouts" / f"rollout_{i}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    (ipc_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def read_input_shard(ipc_dir: Path, rank: int, world_size: int) -> tuple[list[dict], dict[str, Any]]:
    config = json.loads((ipc_dir / "config.json").read_text(encoding="utf-8"))
    paths = sorted((ipc_dir / "rollouts").glob("rollout_*.json"), key=lambda p: int(p.stem.split("_")[1]))[rank::world_size]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths], config


def write_metrics(ipc_dir: Path, metrics: dict[str, Any]) -> None:
    (ipc_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metrics(ipc_dir: Path) -> dict[str, Any]:
    return json.loads((ipc_dir / "metrics.json").read_text(encoding="utf-8"))


def cleanup(ipc_dir: Path) -> None:
    shutil.rmtree(ipc_dir, ignore_errors=True)


def cleanup_run(run_name: str) -> None:
    prefix = ipc_dir_for_step(run_name, 0).name.rsplit("_step_", 1)[0] + "_step_"
    if IPC_ROOT.exists():
        for child in IPC_ROOT.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                shutil.rmtree(child, ignore_errors=True)
