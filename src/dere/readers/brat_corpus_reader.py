import os
from itertools import product
from typing import Optional, Dict, List, Sequence, cast

from nltk.tokenize import TreebankWordTokenizer

from dere.corpus import Corpus, Instance
from dere.corpus import Annotation, SpanAnnotation, FrameAnnotation
from dere.readers import CorpusReader
from dere.schema import SpanType, FrameType

word_tokenizer = TreebankWordTokenizer()


class BRATCorpusReader(CorpusReader):
    def load(self) -> Corpus:
        corpus = Corpus()
        self._populate_corpus(corpus)
        return corpus

    def _populate_corpus(self, corpus: Corpus) -> None:
        doc_id_list = list({
            fname[:-3]
            for fname in os.listdir(self._corpus_path)
            if fname.endswith('.a1')
        })
        for cur_id in doc_id_list:
            annotation2filename = str(self._corpus_path / (cur_id+'.a2'))
            # extend is O(n) in length of second list, not in length of both
            self.read_data(
                corpus=corpus,
                textfilename=str(self._corpus_path / (cur_id+'.txt')),
                doc_id=cur_id,
                annotation1filename=str(self._corpus_path / (cur_id+'.a1')),
                annotation2filename=annotation2filename,
            )

    def read_data(
        self, corpus: Corpus, textfilename: str, doc_id: str,
        annotation1filename: str,
        annotation2filename: Optional[str] = None
    ) -> None:
        annotation_filenames = [annotation1filename]
        if annotation2filename is not None:
            annotation_filenames.append(annotation2filename)

        span_annotations = {}
        frame_annotations = {}

        # First pass -- construct our span annotations, and instantiate
        # our frame annotations

        for filename in annotation_filenames:
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if line[0] == "T":  # Text spans
                        tag, type_begin_end, span_string = line.split("\t")
                        span_type_name, begin, end = type_begin_end.split(" ")
                        span_type = self._schema.type_lookup(
                            "span:" + span_type_name
                        )
                        if type(span_type) is not SpanType:
                            continue
                        span_type = cast(SpanType, span_type)
                        # We are mis-using the left and right fields,
                        # as they should hold the offset in the instance
                        # and not in the file.  We will fix this later.
                        span = SpanAnnotation(
                            span_type,
                            int(begin),
                            int(end),
                            span_string,
                        )
                        span_annotations[tag] = span

                    elif line[0] == "E":  # Events = frames
                        tag, *kvpairs = line.strip().split()
                        frame_type_name, _ = kvpairs[0].rsplit(":", 1)
                        frame_type = self._schema.type_lookup(
                            "frame:" + frame_type_name
                        )
                        if type(frame_type) is not FrameType:
                            continue
                        frame_type = cast(FrameType, frame_type)
                        frame_annotations[tag] = FrameAnnotation(frame_type)

        annotations: Dict[str, Annotation] = {
            **span_annotations, **frame_annotations
        }

        # second pass -- fill the slots for our frames
        for filename in annotation_filenames:
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if line[0] == "E":
                        tag, *kvpairs = line.strip().split()
                        frame_annotation = frame_annotations[tag]
                        for kv in kvpairs:
                            slot_name, filler_tag = kv.rsplit(":", 1)
                            if slot_name in frame_annotation.slots:
                                filler = annotations[filler_tag]
                                frame_annotation.slots[slot_name].add(filler)

        # construct our instances
        with open(textfilename) as f:
            end_offset = 0
            for line in f:
                start_offset = end_offset
                end_offset = start_offset + len(line)
                instance = Instance(line)
                instance_spans = spans_in_window(
                    start_offset, end_offset, list(span_annotations.values())
                )
                for span_annotation in instance_spans:
                    # we mis-used these fields before -- correcting now
                    span_annotation.left -= start_offset
                    span_annotation.right -= start_offset
                instance_frames = frames_referencing_spans(
                    list(frame_annotations.values()), instance_spans
                )
                instance.annotations.extend(instance_spans)
                instance.annotations.extend(instance_frames)
                corpus.instances.append(instance)


def spans_in_window(
    start_window: int, end_window: int, spans: List[SpanAnnotation]
) -> List[SpanAnnotation]:
    window_spans = []
    for a in spans:
        if a.left >= start_window and a.right <= end_window:
            window_spans.append(a)
    return window_spans


def frames_referencing_spans(
    frames: List[FrameAnnotation], target_spans: List[SpanAnnotation]
) -> List[FrameAnnotation]:
    connected_frames: List[FrameAnnotation] = []
    updated = True
    while updated:
        updated = False
        # This is ugly, but mypy doesn't like + for different types
        target: List[Annotation] = []
        target.extend(target_spans)
        target.extend(connected_frames)
        for frame in frames:
            if frame in connected_frames:
                continue
            for slot in frame.slots.values():
                for filler in slot.fillers:
                    if filler in target:
                        connected_frames.append(frame)
                        updated = True
    return list(connected_frames)
