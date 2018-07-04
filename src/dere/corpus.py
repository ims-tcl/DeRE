from typing import List
from dataclasses import dataclass


class Annotation:
    ...


@dataclass
class Instance:
    text: str
    annotations: List[Annotation] = []

        
@dataclass
class Corpus:
    instances: List[Instance] = []


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
