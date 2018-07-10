from __future__ import annotations
from typing import List, Union, Set, Dict
from dataclasses import dataclass, field

from dere.schema import SpanType, FrameType, SlotType


class Annotation:
    ...


@dataclass
class Instance:
    text: str
    annotations: List[Annotation] = field(default_factory=lambda: [])


@dataclass
class Corpus:
    instances: List[Instance] = field(default_factory=lambda: [])


@dataclass
class SpanAnnotation(Annotation):
    span_type: SpanType
    left: int
    right: int
    text: str


@dataclass
class Slot:
    slot_type: SlotType
    fillers: List[Annotation] = field(default_factory=lambda: [])

    def add(self, filler: Annotation) -> None:
        self.fillers.append(filler)


class FrameAnnotation(Annotation):
    def __init__(self, frame_type: FrameType) -> None:
        self.frame_type = frame_type
        self.slots: Dict[str, Slot] = {}
        for slot_type in frame_type.slot_types:
            self.slots[slot_type.name] = Slot(slot_type)
