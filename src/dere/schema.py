from typing import List, Optional, Union


class SpanType:
    def __init__(
        self, name: str,
        anchors: Optional[List[FrameType]] = None
    ) -> None:
        self.name = name
        self.anchors = anchors

    def __repr__(self) -> str:
        return "<Span %s>" % self.name


class FrameType:
    def __init__(self, name: str, slots: List[SlotType]) -> None:
        self.name = name
        self.slots = list(slots)

    def __repr__(self) -> str:
        return "<Frame %s>" % self.name


class SlotType:
    def __init__(
        self, name: str, types: List[Union[FrameType, SpanType]],
        min_cardinality: int = 1, max_cardinality: int = 1
    ) -> None:
        self.name = name
        self.types = types
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

    def __repr__(self) -> str:
        return "<Slot %s>" % self.name


class TaskSchema:
    def __init__(
        self, span_types: List[SpanType],
        frame_types: List[FrameType]
    ) -> None:
        self.span_types = list(span_types)
        self.frame_types = list(frame_types)
