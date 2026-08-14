
import re
from copy import deepcopy

from envs.base import EnvInfo, Observation
from backends.base import BaseLLM
from interaction.spec import load_react_inference_spec


def _restore_prompt_supplied_think_start(content: str | None) -> str:
    text = content or ""
    closing = re.search(r"</think>", text, re.IGNORECASE)
    if closing is None:
        return text
    prefix = text[:closing.start()]
    if re.search(r"<think>", prefix, re.IGNORECASE):
        return text
    leading = text[:len(text) - len(text.lstrip())]
    return leading + "<think>" + text[len(leading):]


def _canonicalize_olmo3_prompt_assistant(content: str | None) -> tuple[dict, str]:
    text = _restore_prompt_supplied_think_start(content)
    closing = re.search(r"</think>", text, re.IGNORECASE)
    if closing is not None:
        search_start = closing.end()
    elif re.match(r"\s*<function_calls>", text, re.IGNORECASE):
        search_start = 0
    else:
        return {"role": "assistant", "content": text}, text

    match = re.search(
        r"<function_calls>\s*(.*?)\s*</function_calls>",
        text[search_start:],
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return {"role": "assistant", "content": text}, text

    absolute_start = search_start + match.start()
    prefix = text[:absolute_start].rstrip()
    body = match.group(1).strip()
    canonical_raw = (
        (prefix + "\n" if prefix else "")
        + "<function_calls>\n"
        + body
        + "\n</function_calls>"
    )
    return {
        "role": "assistant",
        "content": prefix,
        "function_calls": body,
    }, canonical_raw


class ReActAgent:

    def __init__(self, llm: BaseLLM, env_type: str, inference_name: str):
        self.llm = llm
        self.env_type = env_type
        self.inference_name = inference_name
        self._spec = load_react_inference_spec(inference_name)
        self._messages: list[dict] = []
        self._step_count: int = 0
        self._max_steps: int = 20
        self._thinks: list[str | None] = []
        self._latent_embeds = None

    def reset(self, info: EnvInfo, initial_obs: Observation,
              latent_embeds=None) -> None:
        self._max_steps = info.max_steps
        self._step_count = 0
        self._thinks = []
        self._latent_embeds = latent_embeds
        extra = info.extra or {}
        self._tool_result_role = extra.get(
            "prompt_mode_tool_result_role", self._spec.tool_result_role,
        )
        self._restore_prompt_think_start = bool(
            extra.get("restore_prompt_think_start", False)
        )
        self._prompt_mode_assistant_protocol = extra.get(
            "prompt_mode_assistant_protocol"
        )

        user_content = self._spec.build_task_message(info.instruction, initial_obs)

        system_content = extra.get("system_prompt") \
            or self._spec.build_system_prompt(info.env_type)
        system_message = extra.get("system_message")
        if system_message is None:
            system_message = {"role": "system", "content": system_content}
        elif not isinstance(system_message, dict) or system_message.get("role") != "system":
            raise ValueError("EnvInfo.extra['system_message'] must be a system message mapping")
        self._messages = [
            deepcopy(system_message),
            {"role": "user", "content": user_content},
        ]

    async def step(self, obs: Observation) -> dict:
        self._step_count += 1

        if self._step_count > 1:
            if (
                self._tool_result_role == "environment"
                and obs.get("_obs_type") == "tool"
            ):
                self._messages.append({
                    "role": "environment",
                    "content": (obs.get("text") or "").strip() or "(no output)",
                })
            else:
                self._messages.append({
                    "role": "user",
                    "content": self._spec.build_obs_message(obs),
                })

        if self._step_count >= self._max_steps:
            self._thinks.append(None)
            return {"action": "submit"}

        assistant_msg = await self.llm(
            self._messages, latent_embeds=self._latent_embeds,
        )
        action_content = assistant_msg.get("content") or ""

        if self._prompt_mode_assistant_protocol == "olmo3":
            assistant_msg, action_content = _canonicalize_olmo3_prompt_assistant(
                action_content
            )
        else:
            if assistant_msg.get("tool_calls"):
                assistant_msg = {"role": "assistant", "content": action_content}
            if self._restore_prompt_think_start:
                action_content = _restore_prompt_supplied_think_start(action_content)
                assistant_msg = {**assistant_msg, "content": action_content}

        self._messages.append(assistant_msg)

        action, think, has_action = self._spec.parse_assistant_action(
            self.env_type, action_content,
        )
        self._thinks.append(think)
        return action if has_action else {"action": "invalid", "_raw": action_content}

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    @property
    def thinks(self) -> list[str | None]:
        return list(self._thinks)
