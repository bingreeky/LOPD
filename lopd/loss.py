from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_reverse_kl(student_topk_lp: torch.Tensor, teacher_topk_lp: torch.Tensor) -> tuple[torch.Tensor, int]:
    if student_topk_lp.shape != teacher_topk_lp.shape:
        raise ValueError(f"shape mismatch: student {tuple(student_topk_lp.shape)} vs teacher {tuple(teacher_topk_lp.shape)}")
    student = _with_tail_bucket(student_topk_lp)
    teacher = _with_tail_bucket(teacher_topk_lp)
    kl_per_token = F.kl_div(teacher, student, reduction="none", log_target=True).sum(dim=-1)
    return kl_per_token.sum(), int(kl_per_token.numel())


def _with_tail_bucket(topk_lp: torch.Tensor) -> torch.Tensor:
    log_top = torch.logsumexp(topk_lp, dim=-1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([topk_lp, torch.log(-torch.expm1(log_top))], dim=-1)
