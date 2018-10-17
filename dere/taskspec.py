from __future__ import annotations
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SpanType:
    name: str
    predict: bool


@dataclass(frozen=True)
class FrameType:
    name: str
    slot_types: Tuple[SlotType, ...] = field(default_factory=lambda: ())

    def slot_type_lookup(self, name: str) -> Optional[SlotType]:
        for st in self.slot_types:
            if st.name == name:
                return st
        return None

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class SlotType:
    name: str
    types: Tuple[Union[FrameType, SpanType], ...]
    min_cardinality: Optional[int] = 1
    max_cardinality: Optional[int] = 1


@dataclass(frozen=True)
class TaskSpecification:
    span_types: Tuple[SpanType, ...]
    frame_types: Tuple[FrameType, ...]

    def span_type_lookup(self, name: str) -> Optional[SpanType]:
        if name.startswith("span:"):
            name = name[5:]
        for st in self.span_types:
            if st.name == name:
                return st
        return None

    def frame_type_lookup(self, name: str) -> Optional[FrameType]:
        if name.startswith("frame:"):
            name = name[6:]
        for ft in self.frame_types:
            if ft.name == name:
                return ft
        return None

    def type_lookup(self, name: str) -> Optional[Union[SpanType, FrameType]]:
        span_possible = True
        frame_possible = True
        if name.startswith("span:"):
            name = name[5:]
            frame_possible = False
        elif name.startswith("frame:"):
            name = name[6:]
            span_possible = False
        if span_possible:
            for st in self.span_types:
                if st.name == name:
                    return st
        if frame_possible:
            for ft in self.frame_types:
                if ft.name == name:
                    return ft
        return None


# Todo: xml schema validation
def load_from_xml(path: str) -> TaskSpecification:
    tree = ET.parse(path)
    root = tree.getroot()

    # First pass: build our symbol table
    span_types: Dict[str, SpanType] = {}
    frame_types: Dict[str, FrameType] = {}
    symbols: Dict[str, Union[SpanType, FrameType]] = {}
    for child in root.getchildren():
        if child.tag == "spans":
            for spantag in child.getchildren():
                if spantag.tag != "span":
                    continue
                span_name = spantag.attrib["name"]
                predict_string = spantag.attrib["predict"]
                if predict_string == "True":
                    predict = True
                else:
                    predict = False
                span_type = SpanType(span_name, predict)
                span_types[span_name] = span_type
                symbols[span_name] = span_type
                symbols["span:" + span_name] = span_type
        elif child.tag == "frames":
            for frametag in child.getchildren():
                if frametag.tag != "frame":
                    continue
                frame_name = frametag.attrib["name"]
                frame_type = FrameType(frame_name)
                frame_types[frame_name] = frame_type
                symbols[frame_name] = frame_type
                symbols["frame:" + frame_name] = frame_type

    # Second pass -- resolve references
    for child in root.getchildren():
        if child.tag == "spans":
            for spantag in child.getchildren():
                if spantag.tag != "span":
                    continue
                span_name = spantag.attrib["name"]
                span_type = span_types[span_name]
        elif child.tag == "frames":
            for frametag in child.getchildren():
                if frametag.tag != "frame":
                    continue
                frame_name = frametag.attrib["name"]
                slots = []
                for slottag in frametag.getchildren():
                    slot_name = slottag.attrib["name"]
                    slot_type_names = slottag.attrib["types"].split(",")
                    slot_types = tuple(
                        symbols[slot_type_name] for slot_type_name in slot_type_names
                    )
                    min_cardinality = None
                    max_cardinality = None
                    if "mincardinality" in slottag.attrib:
                        min_cardinality = int(slottag.attrib["mincardinality"])
                    if "maxcardinality" in slottag.attrib:
                        max_cardinality = int(slottag.attrib["maxcardinality"])
                    if "cardinality" in slottag.attrib:
                        min_cardinality = int(slottag.attrib["cardinality"])
                        max_cardinality = min_cardinality
                    slot = SlotType(
                        slot_name, slot_types, min_cardinality, max_cardinality
                    )
                    slots.append(slot)
                frame_type = frame_types[frame_name]
                object.__setattr__(frame_type, "slot_types", tuple(slots))
    # now that our symbol table is full, make sure the slot types are right
    return TaskSpecification(tuple(span_types.values()), tuple(frame_types.values()))
