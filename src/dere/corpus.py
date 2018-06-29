from typing import List


class Annotation:
    ...


class Instance:
    def __init__(self, text: str, annotations: List[Annotation]) -> None:
        self.text = text
        self.annotations = []

class Corpus:
    def __init__(self):
        self.instances: List[Instance] = []


class SpanAnnotation(Annotation):
    def __init__(self, left: int, right: int, type_: str) -> None:
        self.left = left
        self.right = right
        self.type_ = type_


class SlotAnnotation(Annotation):
    def __init__(self, type_: str) -> None:
        self.type_ = type_


class FrameAnnotation(Annotation):
    def __init__(self, slots: List[SlotAnnotation]) -> None:
        self.slots = slots


class FrameRepresentation:
    def __init__(self, schema_file: str) -> None:
        ...
