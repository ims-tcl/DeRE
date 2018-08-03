from __future__ import annotations
from typing import List, Union, Set, Dict, Optional
from dataclasses import dataclass, field

from dere.taskspec import SpanType, FrameType, SlotType


class Instance:
    def __init__(self, text: str, corpus: Corpus) -> None:
        self.text = text
        self.corpus = corpus
        self.spans: List[Span] = []
        self.frames: List[Frame] = []

    def new_span(self, span_type: SpanType, left: int, right: int) -> Span:
        span = Span(span_type, left, right, self)
        self.spans.append(span)
        return span

    def new_frame(self, frame_type: FrameType) -> Frame:
        frame = Frame(frame_type, self)
        self.frames.append(frame)
        return frame


class Corpus:
    def __init__(self) -> None:
        self.instances: List[Instance] = []

    def new_instance(self, text: str) -> Instance:
        instance = Instance(text, self)
        self.instances.append(instance)
        return instance


class Span:
    def __init__(
        self,
        span_type: SpanType,
        left: int,
        right: int,
        instance: Instance
    ) -> None:
        self.span_type = span_type
        self.left = left
        self.right = right
        self.instance = instance

    def remove(self) -> None:
        self.instance.spans.remove(self)

    @property
    def text(self):
        return self.instance.text[self.left:self.right]


class Slot:
    def __init__(self, slot_type: SlotType, frame: Frame) -> None:
        self.slot_type = slot_type
        self.frame = frame
        self.fillers: List[Filler] = []

    def add(self, filler: Filler) -> None:
        self.fillers.append(filler)


class Frame:
    def __init__(self, frame_type: FrameType, instance: Optional[Instance]) -> None:
        self.frame_type = frame_type
        self.instance = instance
        self.slots: Dict[SlotType, Slot] = {}
        for slot_type in frame_type.slot_types:
            self.slots[slot_type] = Slot(slot_type, self)

    def remove(self) -> None:
        if self.instance is not None:
            self.instance.frames.remove(self)

    def slot_lookup(self, slot_name: str) -> Optional[Slot]:
        for slot_type, slot in self.slots.items():
            if slot_type.name == slot_name:
                return slot
        return None


Filler = Union[Span, Frame]
