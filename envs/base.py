from abc import ABC, abstractmethod
from dataclasses import dataclass, field


Observation = dict


@dataclass
class EnvInfo:
    task_id: str
    instruction: str
    max_steps: int
    extra: dict = field(default_factory=dict)


class Environment(ABC):

    @abstractmethod
    def get_info(self) -> EnvInfo: ...

    @abstractmethod
    def reset(self, index: int) -> Observation: ...

    @abstractmethod
    def step(self, action: dict) -> tuple[Observation, float, bool, dict]: ...

    def close(self): ...
