from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class SpanType:
    name: str
    anchors: Optional[List[FrameType]] = None


@dataclass
class FrameType:
    name: str
    slots: List[SlotType]


@dataclass
class SlotType:
    name: str
    types: List[Union[FrameType, SpanType]]
    min_cardinality: int = 1
    max_cardinality: int = 1


@dataclass
class TaskSchema:
    span_types: List[SpanType]
    frame_types: List[FrameType]
