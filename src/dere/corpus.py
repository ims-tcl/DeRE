from typing import List
from dataclasses import dataclass, field


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
    left: int
    right: int
    type_: str


@dataclass
class SlotAnnotation(Annotation):
    type_: str


@dataclass
class FrameAnnotation(Annotation):
    slots: List[SlotAnnotation]
