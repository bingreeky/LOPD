from envs.base import EnvInfo, Observation
from backends.base import BaseLLM
from interaction.spec import ReActInferenceSpec


class ReActAgent:

    def __init__(self, llm: BaseLLM, spec: ReActInferenceSpec):
        self.llm = llm
        self._spec = spec
        self._messages: list[dict] = []
        self._step_count: int = 0
        self._max_steps: int = 0
        self._latent_embeds = None

    def reset(self, info: EnvInfo, initial_obs: Observation, latent_embeds=None) -> None:
        self._max_steps = info.max_steps
        self._step_count = 0
        self._latent_embeds = latent_embeds
        self._messages = [
            {"role": "system", "content": info.extra["system_prompt"]},
            {"role": "user", "content": self._spec.build_task_message(info.instruction, initial_obs)},
        ]

    async def step(self, obs: Observation) -> dict:
        self._step_count += 1

        if self._step_count > 1:
            self._messages.append({"role": "user", "content": self._spec.build_obs_message(obs)})

        if self._step_count >= self._max_steps:
            return {"action": "submit"}

        assistant_msg = await self.llm(self._messages, latent_embeds=self._latent_embeds)
        self._messages.append(assistant_msg)
        return self._spec.parse_assistant_action(assistant_msg.get("content") or "")

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)
