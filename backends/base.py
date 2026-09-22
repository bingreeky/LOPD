from abc import ABC, abstractmethod
from typing import Optional

import torch

DEFAULT_MAX_RETRIES = 2


class BaseLLM(ABC):

    @abstractmethod
    async def __call__(
        self,
        messages: list[dict],
        *,
        latent_embeds: Optional[torch.Tensor] = None,
        lora_path: Optional[str] = None,
    ) -> dict: ...
