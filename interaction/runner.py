import asyncio
from dataclasses import dataclass, field

from envs.base import Environment
from interaction.react import ReActAgent


@dataclass
class EpisodeResult:
    task_id: str
    instruction: str
    reward: float
    steps: int
    trajectory: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


async def run_episode(
    env: Environment,
    agent: ReActAgent,
    index: int,
    latent_embeds=None,
) -> EpisodeResult:
    obs = await asyncio.to_thread(env.reset, index)
    info = env.get_info()
    agent.reset(info, obs, latent_embeds=latent_embeds)

    reward = 0.0
    done = False
    step_info = {}
    steps = 0

    for _ in range(info.max_steps):
        action = await agent.step(obs)
        obs, reward, done, step_info = await asyncio.to_thread(env.step, action)
        steps += 1
        if done:
            break

    return EpisodeResult(
        task_id=info.task_id,
        instruction=info.instruction,
        reward=reward,
        steps=steps,
        trajectory=agent.messages,
        info=step_info,
    )
