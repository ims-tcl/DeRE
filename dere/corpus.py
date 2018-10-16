from __future__ import annotations
from typing import List, Union, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field
from itertools import count
import random

import networkx as nx

from dere.taskspec import SpanType, FrameType, SlotType


class Instance:
    """
    This class corresponds to the smallest "unit" of processing in dere.  Every part of the corpus's text
    belongs to exactly one Instance.  Each Span and Frame belongs to exactly one Instance, and all relations
    between Frames and Spans occur between those within the same Instance. This, each Instance is entirely
    self-contained.

    How a Corpus is split into Instances depends on the corpus format and the CorpusIO. It might be that each
    sentence is its own Instance, or that each paragraph is its own Instance, or that each document is its own
    Instance.
    """
    def __init__(self, text: str, document_id: str, corpus: Corpus) -> None:
        """
        Creates a new Instance. This constructor should not be used directly -- instead, Corpus.new_instance()
        should be used to construct new Instances.
        """
        self.text = text
        self.document_id = document_id
        self.corpus = corpus
        self.spans: List[Span] = []
        self.frames: List[Frame] = []
        self.span_indices: Set[int] = set()

    def new_span(
        self,
        span_type: SpanType,
        left: int,
        right: int,
        source: str = 'predicted',
        index: Optional[int] = None
    ) -> Span:
        """
        Construct a new Span, and add it to the Instance. This method should be used instead of calling Span's
        constructor directly.

        Args:
            span_type: The new span's SpanType.
            left: The offset within the Instance text of the first character of the span.
            right: One plus the offset of the last character of the span.
            source: Either 'predicted', 'given', or 'gold'. If 'predicted', the new span is a span predicted
                by a model. If 'given', the new span was present in the input corpus, and is visible to models
                during prediction.  If 'gold', the new span is part of the input corpus's gold-standard label
                set, and should not be visible to models during prediction.
            index: If the span was read from a corpus file, and had some index in that file, this field stores
                that index. This ensures that span indices are preserved between reading and writing.

        Returns:
            The newly created Span.
        """
        span = Span(span_type, left, right, self, source, index)
        self.spans.append(span)
        return span

    def new_frame(self, frame_type: FrameType, source: str = 'predicted') -> Frame:
        """
        Construct a new Frame, and add it to the Instance. This method should be used instead of calling
        Frame's constructor directly.

        The new Frame's slots will all be empty -- these should be filled after the Frame is created.

        Args:
            frame_type: The new frame's FrameType.
            source: Either 'predicted', 'given', or 'gold'. If 'predicted', the new span is a span predicted
                by a model. If 'given', the new span was present in the input corpus, and is visible to models
                during prediction.  If 'gold', the new span is part of the input corpus's gold-standard label
                set, and should not be visible to models during prediction.

        Returns:
            The newly created Span.
        """

        frame = Frame(frame_type, self, source)
        self.frames.append(frame)
        return frame

    def frame_graph(self) -> nx.DiGraph:
        """
        Constructs a directed graph of relations between Frames in the Instance. Each node of the graph
        is a Frame, and each edge from Frame f1 to Frame f2 is the slot in f1 which is filled by f2. Spans are
        not present in this graph.

        Returns:
            The directed graph.
        """
        g = nx.DiGraph()
        for frame in self.frames:
            g.add_node(frame, frame=frame)
            for slot in frame.slots.values():
                for filler in slot.fillers:
                    if isinstance(filler, Frame):
                        g.add_edge(frame, filler, slot=slot)
        return g


class Corpus:
    def __init__(self) -> None:
        """
        Constructs a new Corpus with no Instances.
        """
        self.instances: List[Instance] = []

    def new_instance(self, text: str, document_id: str) -> Instance:
        '''
        Construct a new instance, and add it to the corpus.  This method should be called instead of using
        the constructor for Instance directly, as both Corpus and Instance reference eachother.

        The new Instance will have text, but no spans or frames -- these should be created after the Instance,
        via Instance.new_span() and Instance.new_frame().

        Args:
            text: The instance's text.
            document_id: The document the instance appears in.  The exact meaning of this might depend on the
                corpus format being read, but it usually corresponds to a filename.

        Returns:
            The newly created Instance.
        '''
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
                span2 = instance2.new_span(span.span_type, span.left, span.right, span.source, span.index)
                bijection[span] = span2
            for frame in instance.frames:
                frame2 = instance2.new_frame(frame.frame_type, frame.source)
                bijection[frame] = frame2
            for frame in instance.frames:
                frame2 = bijection[frame]
                for slot_type, slot in frame.slots.items():
                    slot2 = frame2.slots[slot_type]
                    for filler in slot.fillers:
                        slot2.add(bijection[filler])
        return left, right

    # is this elegant or a horrible hack?  You be the judge.
    def clone(self) -> Corpus:
        return self.split(1.0)[0]

    def strip_gold(self) -> None:
        """
        Remove all Spans and Frames whose source is 'gold'
        """
        for instance in self.instances:
            for span in list(instance.spans):
                if span.source == 'gold':
                    span.remove()
            for frame in list(instance.frames):
                if frame.source == 'gold':
                    frame.remove()


class Span:
    """
    A contiguous, span of text within the corpus, labeled by a SpanType.
    """
    def __init__(
        self,
        span_type: SpanType,
        left: int,
        right: int,
        instance: Instance,
        source: str,
        index: Optional[int]
    ) -> None:
        if left > right:
            raise ValueError("Can't create Span: left can't be bigger than right")
        self.span_type = span_type
        self.left = left
        self.right = right
        self.instance = instance
        self.source = source
        self.index = index

    def remove(self) -> None:
        """
        Remove this span from its Instance.
        """
        self.instance.spans.remove(self)

    @property
    def text(self) -> str:
        return self.instance.text[self.left:self.right]

    def matches(self, other: Span) -> bool:
        """
        Checks if this Span is identical to another Span, which might be in another Corpus.

        Args:
            other: The other Span.

        Returns:
            True if the two spans are identical, False otherwise.
        """
        return isinstance(other, Span) and (
            (self.instance.document_id, self.span_type, self.left, self.right)
            ==
            (other.instance.document_id, other.span_type, other.left, other.right)
        )


class Slot:
    def __init__(self, slot_type: SlotType, frame: Frame) -> None:
        self.slot_type = slot_type
        self.frame = frame
        self.fillers: List[Filler] = []

    def add(self, filler: Filler) -> None:
        self.fillers.append(filler)

    def remove(self, filler: Filler) -> None:
        self.fillers.remove(filler)


class Frame:
    def __init__(self, frame_type: FrameType, instance: Optional[Instance], source: str) -> None:
        self.frame_type = frame_type
        self.instance = instance
        self.source = source
        self.slots: Dict[SlotType, Slot] = {}
        for slot_type in frame_type.slot_types:
            self.slots[slot_type] = Slot(slot_type, self)

    def remove(self) -> None:
        if self.instance is not None:
            self.instance.frames.remove(self)
            # remove references to this frame from other frames in the instance
            for frame in self.instance.frames:
                for slot in frame.slots.values():
                    try:
                        slot.remove(self)
                    except ValueError:
                        pass

    def slot_lookup(self, slot_name: str) -> Optional[Slot]:
        for slot_type, slot in self.slots.items():
            if slot_type.name == slot_name:
                return slot
        return None


Filler = Union[Span, Frame]
