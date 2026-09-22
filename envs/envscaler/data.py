from __future__ import annotations

import json
from pathlib import Path
from typing import List


def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_env_metadata(env_meta_path: str) -> dict:
    raw = _load_json(env_meta_path)
    if isinstance(raw, dict):
        records = list(raw.values())
    else:
        records = raw
    by_id = {}
    for e in records:
        rec = dict(e)
        rec["tools"] = _maybe_json(rec.get("tools")) or []
        by_id[rec["env_id"]] = rec
    return by_id


def load_envscaler_samples(
    rl_scenario_path: str,
    env_meta_path: str,
) -> List[dict]:
    env_by_id = load_env_metadata(env_meta_path)
    scenarios = _load_json(rl_scenario_path)
    samples: List[dict] = []
    for t in scenarios:
        env_id = t["env_id"]
        if env_id not in env_by_id:
            raise KeyError(f"RL scenario {t.get('task_id')} references unknown env_id {env_id}")
        env = env_by_id[env_id]
        samples.append({
            "task_id": t["task_id"],
            "env_id": env_id,
            "env_class_name": t["env_class_name"],
            "env_class_code": env["env_class_code"],
            "task": t["task"],
            "init_config": _maybe_json(t["init_config"]) or {},
            "checklist_with_func": t.get("checklist_with_func", []),
            "tools": env["tools"],
            "environment_introduction": env.get("environment_introduction", ""),
            "constraints_rules": env.get("constraints_rules", []),
        })
    return samples
