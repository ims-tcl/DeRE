from __future__ import annotations
from collections import defaultdict
from typing import Union, Collection, List, Dict, Tuple, Any

import networkx as nx

from dere.taskspec import TaskSpecification, SpanType, FrameType
from dere.corpus import Corpus, Instance, Frame, Span

_SFType = Union[SpanType, FrameType]


def evaluate_instances(hypo: List[Instance], gold: List[Instance], task_spec: TaskSpecification) -> Result:
    r = Result(task_spec)
    hypo_spans: List[Span] = []
    hypo_frames: List[Frame] = []
    for instance in hypo:
        hypo_spans += instance.spans
        hypo_frames += instance.frames
    gold_spans: List[Span] = []
    gold_frames: List[Frame] = []
    for instance in gold:
        gold_spans += instance.spans
        gold_frames += instance.frames
    for hspan in hypo_spans:
        for gspan in gold_spans:
            if hspan.matches(gspan):
                r.true_positives[hspan.span_type] += 1
                break
        else:
            r.false_positives[hspan.span_type] += 1
    for gspan in gold_spans:
        for hspan in hypo_spans:
            if hspan.matches(gspan):
                break
        else:
            r.false_negatives[gspan.span_type] += 1

    hgraphs = [instance.frame_graph() for instance in hypo]
    ggraphs = [instance.frame_graph() for instance in gold]

    # two frames are equivalent iff they are members of isomorphic connected components
    # for each gold connected component, see if we can find an isomorphic one in the hypo
    # any leftover hypo connected components contain false positive frames
    hccs = [hgraph.subgraph(c) for hgraph in hgraphs for c in nx.connected_components(nx.Graph(hgraph))]
    gccs = [ggraph.subgraph(c) for ggraph in ggraphs for c in nx.connected_components(nx.Graph(ggraph))]

    # Two frames match if they have the same frame type, and all of their span fillers match
    def node_match(n1: dict, n2: dict) -> bool:
        f1 = n1['frame']
        f2 = n2['frame']
        if f1.frame_type != f2.frame_type:
            return False
        for slot_type in f1.slots:
            if len(f1.slots[slot_type].fillers) != len(f2.slots[slot_type].fillers):
                return False
            for filler1 in f1.slots[slot_type].fillers:
                if isinstance(filler1, Span):
                    for filler2 in f2.slots[slot_type].fillers:
                        if filler1.matches(filler2):
                            break
                    else:
                        return False
        return True

    def edge_match(e1: dict, e2: dict) -> bool:
        slot1 = e1['slot']
        slot2 = e2['slot']
        return bool(slot1.slot_type == slot2.slot_type)

    for gcc in gccs:
        for i, hcc in enumerate(hccs):
            if nx.is_isomorphic(gcc, hcc, node_match=node_match, edge_match=edge_match):
                for frame in gcc.nodes():
                    assert isinstance(frame, Frame)
                    r.true_positives[frame.frame_type] += 1
                del hccs[i]
                break
        else:
            for frame in gcc.nodes():
                assert isinstance(frame, Frame)
                r.false_negatives[frame.frame_type] += 1

    # everything left over in hcc is a false positive
    for hcc in hccs:
        for frame in hcc.nodes():
            assert isinstance(frame, Frame)
            r.false_positives[frame.frame_type] += 1

    return r


def evaluate(hypo: Corpus, gold: Corpus, task_spec: TaskSpecification) -> Result:
    hypo_docs: Dict[str, List[Instance]] = defaultdict(list)
    for hypo_instance in hypo.instances:
        hypo_docs[hypo_instance.document_id].append(hypo_instance)
    gold_docs: Dict[str, List[Instance]] = defaultdict(list)
    for gold_instance in gold.instances:
        gold_docs[gold_instance.document_id].append(gold_instance)
    doc_pairs: List[Tuple[List[Instance], List[Instance]]] = []
    for doc_id in hypo_docs.keys() | gold_docs.keys():
        doc_pairs.append((hypo_docs[doc_id], gold_docs[doc_id]))

    result = Result(task_spec)
    for hypo_instances, gold_instances in doc_pairs:
        result |= evaluate_instances(hypo_instances, gold_instances, task_spec)
    return result


def _string_table(table: List[Union[List[Any], str]], padding: int = 2) -> str:
    column_widths: Dict[int, int] = {}
    for row in table:
        if not isinstance(row, str):
            for i, col in enumerate(row):
                col = str(col)
                if i not in column_widths:
                    column_widths[i] = 0
                column_widths[i] = max(len(col), column_widths[i])
    s = ""
    for row in table:
        if isinstance(row, str):
            line = row
        else:
            line = ""
            for i, col in enumerate(row):
                col = str(col)
                full_width = column_widths[i] + padding
                line += col
                npad = full_width - len(col)
                line += " " * npad
            line = line.rstrip()
        s += line + "\n"
    return s


class Result:
    def __init__(self, task_spec: TaskSpecification) -> None:
        self.task_spec = task_spec
        self.true_positives: Dict[_SFType, int] = defaultdict(int)
        self.false_positives: Dict[_SFType, int] = defaultdict(int)
        self.false_negatives: Dict[_SFType, int] = defaultdict(int)

    def tp(self, cls: Union[_SFType, Collection[_SFType]]) -> int:
        if not isinstance(cls, Collection):
            cls = [cls]
        n = 0
        for c in cls:
            n += self.true_positives[c]
        return n

    def fp(self, cls: Union[_SFType, Collection[_SFType]]) -> int:
        if not isinstance(cls, Collection):
            cls = [cls]
        n = 0
        for c in cls:
            n += self.false_positives[c]
        return n

    def fn(self, cls: Union[_SFType, Collection[_SFType]]) -> int:
        if not isinstance(cls, Collection):
            cls = [cls]
        n = 0
        for c in cls:
            n += self.false_negatives[c]
        return n

    def precision(self, cls: Union[_SFType, Collection[_SFType]]) -> float:
        tp = self.tp(cls)
        if tp == 0:
            return 0
        else:
            return tp / (tp + self.fp(cls))

    def recall(self, cls: Union[_SFType, Collection[_SFType]]) -> float:
        tp = self.tp(cls)
        if tp == 0:
            return 0
        else:
            return tp / (tp + self.fn(cls))

    def fscore(self, cls: Union[_SFType, Collection[_SFType]], beta: float = 1) -> float:
        b2 = beta**2
        if b2 == 0:
            return self.recall(cls)
        elif b2 == float('inf'):
            return self.precision(cls)
        precision = self.precision(cls)
        recall = self.recall(cls)
        if precision == 0 or recall == 0:
            return 0
        return (1+b2) / (b2/precision + 1/recall)

    def __or__(self, other: Result) -> Result:
        assert self.task_spec == other.task_spec
        r = Result(self.task_spec)
        sftypes: Tuple[_SFType, ...] = self.task_spec.span_types
        sftypes += self.task_spec.frame_types
        for sf_type in sftypes:
            r.true_positives[sf_type] = self.true_positives[sf_type] + other.true_positives[sf_type]
            r.false_positives[sf_type] = self.false_positives[sf_type] + other.false_positives[sf_type]
            r.false_negatives[sf_type] = self.false_negatives[sf_type] + other.false_negatives[sf_type]
        return r

    def report(self) -> str:
        def row(cls: Union[_SFType, Collection[_SFType]], label: str) -> List[Any]:
            tp = self.tp(cls)
            fp = self.fp(cls)
            fn = self.fn(cls)
            return [
                label,
                tp+fn,
                tp+fp,
                tp,
                "%.2f" % (100*self.recall(cls)),
                "%.2f" % (100*self.precision(cls)),
                "%.2f" % (100*self.fscore(cls))
            ]
        table: List[Union[List[Any], str]] = [[
            "Class", "gold", "answer", "match", "recall", "prec.", "fscore"
        ]]
        table.append("-------------- SPAN EVALUATION ------------------")
        for span_type in self.task_spec.span_types:
            table.append(row(span_type, span_type.name))
        table.append(row(self.task_spec.span_types, "=[SPAN TOTAL]="))
        table.append('----------------------------------------------')
        table.append('-------------- FRAME EVALUATION ------------------')
        for frame_type in self.task_spec.frame_types:
            table.append(row(frame_type, frame_type.name))
        table.append(row(self.task_spec.frame_types, "=[FRAME TOTAL]="))
        table.append('----------------------------------------------')
        all_types: Tuple[_SFType, ...] = self.task_spec.span_types
        all_types += self.task_spec.frame_types
        table.append(row(all_types, "=[TOTAL]="))
        return _string_table(table)
