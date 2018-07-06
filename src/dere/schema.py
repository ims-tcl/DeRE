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
    slots: List[SlotType] = field(default_factory=lambda: [])


@dataclass
class SlotType:
    name: str
    types: List[Union[FrameType, SpanType]]
    min_cardinality: Optional[int] = 1
    max_cardinality: Optional[int] = 1


@dataclass
class TaskSchema:
    span_types: List[SpanType]
    frame_types: List[FrameType]


# Todo: xml schema validation
def load_task_schema_file(path: str) -> TaskSchema:
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
                frame_type.slots = slots
    # now that our symbol table is full, make sure the slot types are right
    return TaskSchema(list(span_types.values()), list(frame_types.values()))
