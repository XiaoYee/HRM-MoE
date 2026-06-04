from __future__ import annotations

import os
from typing import Optional

import torch


_MOE_PROFILE_ENABLED = os.environ.get("HRM_MOE_PROFILE", "0").lower() in ("1", "true", "yes")
_MOE_PROFILE_SECONDS: dict[str, float] = {}


class MoEProfilePhase:
    def __init__(self, name: str):
        self.name = name
        self.start: Optional[torch.cuda.Event] = None
        self.end: Optional[torch.cuda.Event] = None

    def __enter__(self):
        if not _MOE_PROFILE_ENABLED or not torch.cuda.is_available():
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.start is None or self.end is None:
            return False
        self.end.record()
        self.end.synchronize()
        elapsed = self.start.elapsed_time(self.end) / 1000.0
        _MOE_PROFILE_SECONDS[self.name] = _MOE_PROFILE_SECONDS.get(self.name, 0.0) + elapsed
        return False


def reset_moe_profile() -> None:
    _MOE_PROFILE_SECONDS.clear()


def pop_moe_profile() -> dict[str, float]:
    profile = dict(_MOE_PROFILE_SECONDS)
    _MOE_PROFILE_SECONDS.clear()
    return profile


def record_moe_profile_phase(name: str) -> MoEProfilePhase:
    return MoEProfilePhase(name)
