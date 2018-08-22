from __future__ import annotations
from typing import List, Union, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field
from itertools import count
import random

from dere.taskspec import SpanType, FrameType, SlotType


class Instance:
    def __init__(self, text: str, document_id: str, corpus: Corpus) -> None:
        self.text = text
        self.document_id = document_id
        self.corpus = corpus
        self.spans: List[Span] = []
        self.frames: List[Frame] = []
        self.span_indices: Set[int] = set()

    def new_span(self, span_type: SpanType, left: int, right: int, index: Optional[int] = None) -> Span:
        span = Span(span_type, left, right, self, index)
        self.spans.append(span)
        return span

    def new_frame(self, frame_type: FrameType) -> Frame:
        frame = Frame(frame_type, self)
        self.frames.append(frame)
        return frame


class Corpus:
    def __init__(self) -> None:
        self.instances: List[Instance] = []

    def new_instance(self, text: str, document_id: str) -> Instance:
        instance = Instance(text, document_id, self)
        self.instances.append(instance)
        return instance

    def split(self, ratio: float) -> Tuple[Corpus, Corpus]:
        left = Corpus()
        right = Corpus()
        for instance in self.instances:
            if random.random() < ratio:
                corpus2 = left
            else:
                corpus2 = right
            instance2 = corpus2.new_instance(instance.text, instance.document_id)
            bijection: Dict[Filler, Filler] = {}
            for span in instance.spans:
                span2 = instance2.new_span(span.span_type, span.left, span.right)
                bijection[span] = span2
            for frame in instance.frames:
                frame2 = instance2.new_frame(frame.frame_type)
                bijection[frame] = frame2
            for frame in instance.frames:
                frame2 = bijection[frame]
                for slot_type, slot in frame.slots.items():
                    slot2 = frame2.slots[slot_type]
                    for filler in slot.fillers:
                        slot2.add(bijection[filler])
        return left, right

    # is this elegant or a horrible hack?  You be the judge.
    def clone(self):
        return self.split(1.0)[0]


class Span:
    def __init__(
        self,
        span_type: SpanType,
        left: int,
        right: int,
        instance: Instance,
        index: Optional[int]
    ) -> None:
        if left > right:
            raise ValueError("Can't create Span: left can't be bigger than right")
        self.span_type = span_type
        self.left = left
        self.right = right
        self.instance = instance
        self.index = index

    def remove(self) -> None:
        self.instance.spans.remove(self)

    @property
    def text(self) -> str:
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
