from __future__ import annotations
from typing import List, Optional, Union, Dict
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET


@dataclass
class SpanType:
    name: str
    anchors: List[FrameType] = field(default_factory=lambda: [])


@dataclass
class FrameType:
    name: str
    slot_types: List[SlotType] = field(default_factory=lambda: [])

    def slot_type_lookup(self, name: str) -> Optional[SlotType]:
        for st in self.slot_types:
            if st.name == name:
                return st
        return None


@dataclass
class SlotType:
    name: str
    types: List[Union[FrameType, SpanType]]
    min_cardinality: Optional[int] = 1
    max_cardinality: Optional[int] = 1


@dataclass
class TaskSpecification:
    span_types: List[SpanType]
    frame_types: List[FrameType]

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
        if child.tag == 'spantypes':
            for spantag in child.getchildren():
                if spantag.tag != 'span':
                    continue
                span_name = spantag.attrib['name']
                span_type = SpanType(span_name)
                span_types[span_name] = span_type
                symbols[span_name] = span_type
                symbols["span:" + span_name] = span_type
        elif child.tag == 'frames':
            for frametag in child.getchildren():
                if frametag.tag != 'frame':
                    continue
                frame_name = frametag.attrib['name']
                frame_type = FrameType(frame_name)
                frame_types[frame_name] = frame_type
                symbols[frame_name] = frame_type
                symbols["frame:" + frame_name] = frame_type

    # Second pass -- resolve references
    for child in root.getchildren():
        if child.tag == 'spantypes':
            for spantag in child.getchildren():
                if spantag.tag != 'span':
                    continue
                span_name = spantag.attrib['name']
                if 'anchors' in spantag.attrib:
                    anchor_names = spantag.attrib['anchors'].split(',')
                    anchors = [
                        frame_types[anchor_name]
                        for anchor_name in anchor_names
                    ]
                else:
                    anchors = []
                span_type = span_types[span_name]
                span_type.anchors = anchors
        elif child.tag == 'frames':
            for frametag in child.getchildren():
                if frametag.tag != 'frame':
                    continue
                frame_name = frametag.attrib['name']
                slots = []
                for slottag in frametag.getchildren():
                    slot_name = slottag.attrib['name']
                    slot_type_names = slottag.attrib['types'].split(',')
                    slot_types = [
                        symbols[slot_type_name]
                        for slot_type_name in slot_type_names
                    ]
                    min_cardinality = None
                    max_cardinality = None
                    if 'mincardinality' in slottag.attrib:
                        min_cardinality = int(slottag.attrib['mincardinality'])
                    if 'maxcardinality' in slottag.attrib:
                        max_cardinality = int(slottag.attrib['maxcardinality'])
                    if 'cardinality' in slottag.attrib:
                        min_cardinality = int(slottag.attrib['cardinality'])
                        max_cardinality = min_cardinality
                    slot = SlotType(
                        slot_name, slot_types,
                        min_cardinality, max_cardinality
                    )
                    slots.append(slot)
                frame_type = frame_types[frame_name]
                frame_type.slot_types = slots
    # now that our symbol table is full, make sure the slot types are right
    return TaskSpecification(
        list(span_types.values()), list(frame_types.values())
    )
