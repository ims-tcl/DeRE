from __future__ import annotations
from typing import List, Union, Set, Dict, Optional
from dataclasses import dataclass, field

from dere.taskspec import SpanType, FrameType, SlotType


@dataclass
class Instance:
    text: str
    spans: List[Span] = field(default_factory=lambda: [])
    frames: List[Frame] = field(default_factory=lambda: [])


@dataclass
class Corpus:
    instances: List[Instance] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class Span:
    span_type: SpanType
    left: int
    right: int
    text: str


@dataclass
class Slot:
    slot_type: SlotType
    fillers: List[Filler] = field(default_factory=lambda: [])

    def add(self, filler: Filler) -> None:
        self.fillers.append(filler)


class Frame:
    def __init__(self, frame_type: FrameType) -> None:
        self.frame_type = frame_type
        self.slots: Dict[SlotType, Slot] = {}
        for slot_type in frame_type.slot_types:
            self.slots[slot_type] = Slot(slot_type)

    def slot_lookup(self, slot_name: str) -> Optional[Slot]:
        for slot_type, slot in self.slots.items():
            if slot_type.name == slot_name:
                return slot
        return None


Filler = Union[Span, Frame]
