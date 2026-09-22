from __future__ import annotations

import traceback
from typing import List, Optional

from envs.base import EnvInfo, Environment, Observation
from envs.envscaler import runtime as esr

ENV_TYPE = "envscaler"


class EnvScalerEnv(Environment):

    def __init__(
        self,
        samples: List[dict],
        *,
        max_steps: int = 30,
    ):
        self.samples = samples
        self.max_steps = max_steps
        self._sample: Optional[dict] = None
        self.env_instance = None
        self.init_state: dict = {}
        self.checklist_with_func: List[dict] = []
        self._system_prompt: str = ""

    def reset(self, index: int) -> Observation:
        sample = self.samples[index]
        self._sample = sample
        self.checklist_with_func = sample.get("checklist_with_func", []) or []

        env_class = esr.init_env_class(sample["env_class_code"], sample["env_class_name"])
        self.env_instance = esr.init_env_instance(env_class, sample.get("init_config") or {})
        self.init_state = esr.get_state_info(self.env_instance)

        self._system_prompt = esr.build_envscaler_system_prompt(
            sample.get("environment_introduction", ""),
            sample.get("constraints_rules", []),
            sample.get("tools", []),
        )
        return {"text": sample["task"], "success": True, "_obs_type": "user"}

    def get_info(self) -> EnvInfo:
        if self._sample is None:
            raise RuntimeError("Call reset(index) before get_info().")
        return EnvInfo(
            task_id=self._sample["task_id"],
            instruction=self._sample["task"],
            max_steps=self.max_steps,
            extra={"system_prompt": self._system_prompt},
        )

    def step(self, action: dict) -> tuple[Observation, float, bool, dict]:
        if self.env_instance is None:
            raise RuntimeError("Call reset(index) before step().")

        if action.get("action") == "submit":
            env_action = {"name": "chat_with_user", "arguments": {"content": "Task Completed"}}
        else:
            raw = action.get("_raw_text", "")
            parse_ok, struct = esr.parse_response(raw)
            if not parse_ok:
                return self._user_obs(f"Error: malformed response. {struct.get('tool_calls') or struct.get('reasoning_content')}"), \
                    0.0, False, {"action_executed": False, "parse_error": True}
            ok, env_action = esr.parse_action(struct)
            if not ok:
                return self._user_obs("Error: could not parse a valid action."), \
                    0.0, False, {"action_executed": False, "parse_error": True}

        name = env_action.get("name")
        args = env_action.get("arguments", {}) or {}

        if name == "chat_with_user":
            final_state = esr.get_state_info(self.env_instance)
            reward = esr.calculate_reward(self.checklist_with_func, self.init_state, final_state)
            return ({"text": "Task finished", "success": True, "_obs_type": "user"},
                    reward, True, {"action_executed": True, "final": True, "reward": reward})

        if not isinstance(args, dict) or not hasattr(self.env_instance, name):
            return self._user_obs(f"Error: '{name}' is not a valid tool of this environment."), \
                0.0, False, {"action_executed": False, "invalid_action": True}

        try:
            result = getattr(self.env_instance, name)(**args)
        except Exception:
            err = traceback.format_exc()
            return self._user_obs("Error: <Exception>\n" + err), \
                0.0, True, {"action_executed": False, "exception": True}
        return ({"text": f"{result}", "success": True, "_obs_type": "tool"},
                0.0, False, {"action_executed": True, "tool": name})

    @staticmethod
    def _user_obs(text: str) -> Observation:
        return {"text": text, "success": False, "_obs_type": "user"}
