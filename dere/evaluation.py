from __future__ import annotations
from collections import defaultdict
from typing import Union, Collection, List, Dict, Tuple, Any
from functools import total_ordering

import networkx as nx

from dere.taskspec import TaskSpecification, SpanType, FrameType
from dere.corpus import Corpus, Instance, Frame, Span

_SFType = Union[SpanType, FrameType]


def _evaluate_document(hypo: List[Instance], gold: List[Instance], task_spec: TaskSpecification) -> Result:
    r = Result(task_spec)
    hypo_spans: List[Span] = []
    for instance in hypo:
        hypo_spans += [span for span in instance.spans if span.source != 'given']
    gold_spans: List[Span] = []
    for instance in gold:
        gold_spans += [span for span in instance.spans if span.source != 'given']
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
                    if frame.source != 'given':
                        r.true_positives[frame.frame_type] += 1
                del hccs[i]
                break
        else:
            for frame in gcc.nodes():
                assert isinstance(frame, Frame)
                if frame.source != 'given':
                    r.false_negatives[frame.frame_type] += 1

    # everything left over in hcc is a false positive
    for hcc in hccs:
        for frame in hcc.nodes():
            assert isinstance(frame, Frame)
            if frame.source != 'given':
                r.false_positives[frame.frame_type] += 1

    return r


def evaluate(hypo: Corpus, gold: Corpus, task_spec: TaskSpecification) -> Result:
    """
    Evaluate a corpus containing model predictions against a corpus containing gold-standard annotations
    according to a supplied task-spec.

    Args:
        hypo: The hypothesis corpus, containing model predictions.
        gold: The corpus containing gold-standard annotation data.

    Returns:
        A Result object representing the evaluation results.
    """

    # TODO(Sean): Maybe supply a single corpus object, which contains both predicted and gold frames and
    # spans, but with different .source attributes?  Revisit this later to see if this would be a good
    # idea

    # TODO(Sean): Add support for different styles of soft evaluation
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
        result |= _evaluate_document(hypo_instances, gold_instances, task_spec)
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


@total_ordering
class Result:
    """
    Result objects represent evaluation results. They keep track of true positives, false positives, and false
    negatives for frame and span evaluation, and can thus compute precision, recall, and fscore of a given
    evaluation. They are also comparable, and so the results of two evaluations can be compared -- The default
    comparison metric is f1 score on frame evaluation, with f1-score on span evaluation as a fallback.
    """
    def __init__(self, task_spec: TaskSpecification) -> None:
        """
        Create an empty result with no true positives, false positives, or false negatives.

        Args:
            task_spec: The TaskSpecification of the model being evaluated
        """
        self.task_spec = task_spec
        self.true_positives: Dict[_SFType, int] = defaultdict(int)
        self.false_positives: Dict[_SFType, int] = defaultdict(int)
        self.false_negatives: Dict[_SFType, int] = defaultdict(int)

    def tp(self, cls: Union[_SFType, Collection[_SFType]]) -> int:
        """
        Count true positives for a SpanType, FrameType, or collection thereof.

        Args:
            cls: Either the SpanType or FrameType to count true positives for, or a collection of SpanTypes
            and FrameTypes, in which case counts for all elements are summed.

        Returns:
            The number of true positives.
        """
        if not isinstance(cls, Collection):
            cls = [cls]
        n = 0
        for c in cls:
            n += self.true_positives[c]
        return n

    def fp(self, cls: Union[_SFType, Collection[_SFType]]) -> int:
        """
        Count false positives for a SpanType, FrameType, or collection thereof.

        Args:
            cls: Either the SpanType or FrameType to count false positives for, or a collection of SpanTypes
                and FrameTypes, in which case counts for all elements are summed.

        Returns:
            The number of false positives.
        """

        if not isinstance(cls, Collection):
            cls = [cls]
        n = 0
        for c in cls:
            n += self.false_positives[c]
        return n

    def fn(self, cls: Union[_SFType, Collection[_SFType]]) -> int:
        """
        Count false negatives for a SpanType, FrameType, or collection thereof.

        Args:
            cls: Either the SpanType or FrameType to count false negatives for, or a collection of SpanTypes
                and FrameTypes, in which case counts for all elements are summed.

        Returns:
            The number of false negatives.
        """
        if not isinstance(cls, Collection):
            cls = [cls]
        n = 0
        for c in cls:
            n += self.false_negatives[c]
        return n

    def precision(self, cls: Union[_SFType, Collection[_SFType]]) -> float:
        """
        Compute precision for a SpanType of FrameType, or micro-averaged precision for a collection thereof.

        Args:
            cls: Either a SpanType or FrameType, or a collection of SpanTypes and FrameTypes, in which case
                micro-averaged precision will be computed.

        Returns:
            The precision, or zero if there were no true positives and no false positives
        """
        tp = self.tp(cls)
        if tp == 0:
            return 0
        else:
            return tp / (tp + self.fp(cls))

    def recall(self, cls: Union[_SFType, Collection[_SFType]]) -> float:
        """
        Compute recall for a SpanType of FrameType, or micro-averaged recall for a collection thereof.

        Args:
            cls: Either a SpanType or FrameType, or a collection of SpanTypes and FrameTypes, in which case
                micro-averaged recall will be computed.

        Returns:
            The recall, or zero if there were no true positives and no false negatives
        """

        tp = self.tp(cls)
        if tp == 0:
            return 0
        else:
            return tp / (tp + self.fn(cls))

    def fscore(self, cls: Union[_SFType, Collection[_SFType]], beta: float = 1) -> float:
        """
        Compute the :math:`F_{\beta}` score for a SpanType of FrameType, or the micro-averaged
        :math:`F_{\beta}` score for a collection thereof.

        Args:
            cls: Either a SpanType or FrameType, or a collection of SpanTypes and FrameTypes, in which case
                the micro-averaged :math:`F_{\beta}` score will be computed.
            beta: The beta parameter.

        Returns:
            The precision, or zero if there were no true positives and no false positives
        """

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

    def union(self, other: Result) -> Result:
        """
        Returns the union of this Result and another Result.  True positive, false positive, and false
        negative cocunts of the union are just the sum of this Result's and the other's. Neither original
        result object is modified.

        Args:
            other: The other Result with which to take the union.

        Returns:
            The union of this Result and the other Result.
        """
        assert self.task_spec == other.task_spec
        r = Result(self.task_spec)
        sftypes: Tuple[_SFType, ...] = self.task_spec.span_types
        sftypes += self.task_spec.frame_types
        for sf_type in sftypes:
            r.true_positives[sf_type] = self.true_positives[sf_type] + other.true_positives[sf_type]
            r.false_positives[sf_type] = self.false_positives[sf_type] + other.false_positives[sf_type]
            r.false_negatives[sf_type] = self.false_negatives[sf_type] + other.false_negatives[sf_type]
        return r

    def __or__(self, other: Result) -> Result:
        return self.union(other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Result):
            return False
        return (
            self.fscore(self.task_spec.frame_types) == other.fscore(other.task_spec.frame_types)
            and self.fscore(self.task_spec.span_types) == other.fscore(other.task_spec.span_types)
        )

    def __lt__(self, other: Result) -> bool:
        sf = self.fscore(self.task_spec.frame_types)
        of = other.fscore(other.task_spec.frame_types)
        if sf == of:
            return self.fscore(self.task_spec.span_types) < other.fscore(other.task_spec.span_types)
        else:
            return sf < of

    def report(self) -> str:
        """
        Return a pretty-printed report, showing statistics for each SpanType and FrameType.

        Returns:
            A string containing the report, rendered as an ascii-art table.
        """
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
