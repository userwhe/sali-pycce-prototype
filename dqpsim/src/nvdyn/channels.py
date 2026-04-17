"""Drive-channel definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChannelKind = Literal["mw", "rf"]


@dataclass(slots=True, frozen=True)
class DriveChannel:
    """A named microwave or radio-frequency control channel."""

    name: str
    kind: ChannelKind
    targets: tuple[str, ...]


def mw_channel(*, name: str = "mw", targets: tuple[str, ...] = ("electron",)) -> DriveChannel:
    """Create a microwave drive channel."""

    return DriveChannel(name=name, kind="mw", targets=targets)


def rf_channel(*, target: str, name: str | None = None) -> DriveChannel:
    """Create a radio-frequency drive channel for one carbon."""

    return DriveChannel(name=name or f"rf:{target}", kind="rf", targets=(target,))
