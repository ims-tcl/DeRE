from typing import List


class SpanType:
    def __init__(self, name: str, anchors=None) -> None:
        self.name = name
        self.anchors = anchors

    def __repr__(self):
        return "<Span %s>" % self.name


class FrameType:
    def __init__(self, name, slots) -> None:
        self.name = name
        self.slots = list(slots)

    def __repr__(self):
        return "<Frame %s>" % self.name


class SlotType:
    def __init__(self, name, types, min_cardinality=1, max_cardinality=1) -> None:
        self.name = name
        self.types = types
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

    def __repr__(self):
        return "<Slot %s>" % self.name


class TaskSchema:
    def __init__(self, span_types: List[SpanType], frame_types) -> None:
        self.span_types = list(span_types)
        self.frame_types = list(frame_types)


def parse_schema(path):
    tree = ET.parse(path)
    root = tree.getroot()
    span_types = []
    frame_types = []
    symbols = {}
    for child in root.getchildren():
        if child.tag == "spantypes":
            for spantag in child.getchildren():
                if spantag.tag != "span":
                    continue
                name = spantag.attrib["name"]
                if "anchors" in spantag.attrib:
                    anchors = spantag.attrib["anchors"]
                else:
                    anchors = None
                span_type = SpanType(name, anchors)
                span_types.append(span_type)
                symbols[name] = span_type
        elif child.tag == "frames":
            for frametag in child.getchildren():
                if frametag.tag != "frame":
                    continue
                frame_name = frametag.attrib["name"]
                slots = []
                for slottag in frametag.getchildren():
                    slot_name = slottag.attrib["name"]
                    slot_types = slottag.attrib["types"].split(",")
                    min_cardinality = None
                    max_cardinality = None
                    if "mincardinality" in slottag.attrib:
                        min_cardinality = int(slottag.attrib["mincardinality"])
                    if "maxcardinality" in slottag.attrib:
                        max_cardinality = int(slottag.attrib["maxcardinality"])
                    if "cardinality" in slottag.attrib:
                        min_cardinality = int(slottag.attrib["cardinality"])
                        max_cardinality = min_cardinality
                    slot = Slot(slot_name, slot_types, min_cardinality, max_cardinality)
                    slots.append(slot)
                frame_type = FrameType(frame_name, slots)
                symbols[frame_name] = frame_type
                frame_types.append(frame_type)
    # now that our symbol table is full, make sure the slot types are right
    for frame_type in frame_types:
        for slot in frame_type.slots:
            slot.types = [symbols[t] for t in slot.types]
    for span_type in span_types:
        if span_type.anchors is not None:
            span_type.anchors = symbols[span_type.anchors]
    return TaskSchema(span_types, frame_types)
