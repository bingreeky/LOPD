
from __future__ import annotations

from copy import deepcopy
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
        tool_protocol: str = "qwen3",
    ):
        self.samples = samples
        self.max_steps = max_steps
        self.tool_protocol = esr.normalize_tool_protocol(tool_protocol)
        self._sample: Optional[dict] = None
        self._index: Optional[int] = None
        self.env_instance = None
        self.init_state: dict = {}
        self.checklist_with_func: List[dict] = []
        self._system_prompt: str = ""
        self._system_message: dict = {"role": "system", "content": ""}

    def __len__(self) -> int:
        return len(self.samples)

    def reset(self, index: int) -> Observation:
        sample = self.samples[index]
        self._sample = sample
        self._index = index
        self.checklist_with_func = sample.get("checklist_with_func", []) or []

        env_class = esr.init_env_class(sample["env_class_code"], sample["env_class_name"])
        self.env_instance = esr.init_env_instance(env_class, sample.get("init_config") or {})
        self.init_state = esr.get_state_info(self.env_instance)

        self._system_message = esr.build_envscaler_system_message(
            sample.get("environment_introduction", ""),
            sample.get("constraints_rules", []),
            sample.get("tools", []),
            tool_protocol=self.tool_protocol,
        )
        self._system_prompt = self._system_message["content"]
        return {"text": sample["task"], "success": True,
                "_env_type": ENV_TYPE, "_obs_type": "user"}

    def get_info(self) -> EnvInfo:
        if self._sample is None:
            raise RuntimeError("Call reset(index) before get_info().")
        s = self._sample
        extra = {
            "system_prompt": self._system_prompt,
            "env_id": s["env_id"],
            "tool_protocol": self.tool_protocol,
        }
        if self.tool_protocol == "olmo3":
            extra.update({
                "system_message": deepcopy(self._system_message),
                "prompt_mode_tool_result_role": "environment",
                "restore_prompt_think_start": True,
                "prompt_mode_assistant_protocol": "olmo3",
            })
        return EnvInfo(
            task_id=s["task_id"],
            instruction=s["task"],
            env_type=ENV_TYPE,
            max_steps=self.max_steps,
            extra=extra,
        )

    def step(self, action: dict) -> tuple[Observation, float, bool, dict]:
        if self.env_instance is None:
            raise RuntimeError("Call reset(index) before step().")

        if action.get("action") == "submit":
            env_action = {"name": "chat_with_user", "arguments": {"content": "Task Completed"}}
        else:
            raw = action.get("_raw_text", "")
            parse_ok, struct = esr.parse_response(raw, tool_protocol=self.tool_protocol)
            if not parse_ok:
                return self._error_obs(f"Error: malformed response. {struct.get('tool_calls') or struct.get('reasoning_content')}"), \
                    0.0, False, {"action_executed": False, "parse_error": True}
            ok, env_action = esr.parse_action(struct)
            if not ok:
                return self._error_obs("Error: could not parse a valid action."), \
                    0.0, False, {"action_executed": False, "parse_error": True}

        name = env_action.get("name")
        args = env_action.get("arguments", {}) or {}

        if name == "chat_with_user":
            final_state = esr.get_state_info(self.env_instance)
            reward = esr.calculate_reward(self.checklist_with_func, self.init_state, final_state)
            return ({"text": "Task finished", "success": True, "_env_type": ENV_TYPE, "_obs_type": "user"},
                    reward, True, {"action_executed": True, "final": True, "reward": reward})

        if not isinstance(args, dict) or not hasattr(self.env_instance, name):
            return self._user_obs(f"Error: '{name}' is not a valid tool of this environment."), \
                0.0, False, {"action_executed": False, "invalid_action": True}

        try:
            result = getattr(self.env_instance, name)(**args)
        except Exception:  # noqa: BLE001 - faithful to EnvScaler (terminate on exec error)
            err = traceback.format_exc()
            return self._user_obs("Error: <Exception>\n" + err), \
                0.0, True, {"action_executed": False, "exception": True}
        return ({"text": f"{result}", "success": True, "_env_type": ENV_TYPE, "_obs_type": "tool"},
                0.0, False, {"action_executed": True, "tool": name})

    def _error_obs(self, text: str) -> Observation:
        if self.tool_protocol == "olmo3":
            return {
                "text": text, "success": False,
                "_env_type": ENV_TYPE, "_obs_type": "tool",
            }
        return self._user_obs(text)

    @staticmethod
    def _user_obs(text: str) -> Observation:
        return {"text": text, "success": False, "_env_type": ENV_TYPE, "_obs_type": "user"}
