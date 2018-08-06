import os
from os import PathLike
import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Union

from dere.corpus_io import CorpusIO
from dere.corpus import Corpus, Instance, Span, Frame, Filler
from dere.taskspec import TaskSpecification, SpanType


class CQSACorpusIO(CorpusIO):
    def load(self, path: str, load_gold: bool = True) -> Corpus:
        if os.path.isdir(path):
            paths: List[str] = []
            for dirpath, dirnames, filenames in os.walk(path):
                paths.extend(
                    [
                        os.path.join(dirpath, fn)
                        for fn in filenames
                        if fn.endswith(".xml")
                    ]
                )
        else:
            paths = [path]
        corpus = Corpus()
        for path in paths:
            self._populate_corpus_from_file(corpus, path)
        return corpus

    def _populate_corpus_from_file(self, corpus: Corpus, path: str) -> None:
        tree = ET.parse(path)
        root = tree.getroot()
        corpus.instances.append(self._construct_instance(corpus, root))
        """
        for child in root.getchildren():
            if child.tag in ["HEADING", "PARAGRAPH"]:
                instance = self._construct_instance(child)
                corpus.instances.append(instance)
        """

    def _construct_instance(self, corpus: Corpus, element: ET.Element) -> Instance:
        instance = corpus.new_instance("")
        ids: Dict[str, Filler] = {}
        self._populate_instance(element, instance, ids)
        instance.text = instance.text.replace("\n", " ")
        self._link_instance(element, instance, ids)
        return instance

    def _populate_instance(
        self, element: ET.Element, instance: Instance, ids: Dict[str, Filler]
    ) -> None:
        if ids is None:
            ids = {}
        if element.text is not None:
            instance.text += element.text
        for child in element.getchildren():
            left = len(instance.text)
            self._populate_instance(child, instance, ids)
            right = len(instance.text)
            span = None
            span_type = self._spec.span_type_lookup(child.tag)
            if span_type is not None:
                span = instance.new_span(span_type, left, right)
                instance.spans.append(span)
                ids[child.attrib["id"]] = span
            frame_type = self._spec.frame_type_lookup(child.tag)
            if frame_type is not None:
                frame = instance.new_frame(frame_type)
                if span is not None:
                    slot = frame.slot_lookup(frame_type.name)
                    if slot is not None:
                        slot.add(span)
                instance.frames.append(frame)
                ids[child.attrib["id"]] = frame
            if child.tail is not None:
                instance.text += child.tail

    def _link_instance(
        self, element: ET.Element, instance: Instance, ids: Dict[str, Filler]
    ) -> None:
        for element in element.iter():
            if "id" in element.attrib and element.attrib["id"] in ids:
                frame = ids[element.attrib["id"]]
                if isinstance(frame, Frame):
                    for attrib, value in element.attrib.items():
                        slot = frame.slot_lookup(attrib)
                        if slot is not None:
                            filler = ids[value]
                            slot.add(filler)
