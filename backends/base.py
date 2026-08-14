from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):

    @abstractmethod
    async def __call__(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> dict: ...
