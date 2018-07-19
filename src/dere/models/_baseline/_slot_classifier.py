# Author: Laura
import sys
import pdb
import copy
import logging
from itertools import chain, combinations, product
from operator import mul
from typing import Optional, Dict, Tuple, List, Set, Any

import networkx as nx
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.pipeline import DependencyParser
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.externals import joblib
import random
from scipy.sparse import hstack, csr_matrix
import scipy

from dere import Result
from dere.corpus import Corpus, Instance, Frame, Span, Filler
from dere.taskspec import TaskSpecification, FrameType, SpanType, SlotType

nlp = spacy.load("en")


class SlotClassifier:
    def __init__(self, spec: TaskSpecification) -> None:
        self.logger = logging.getLogger(__name__)
        self._spec = spec
        random.seed(98765)
        # Find our plausible relations from the spec
        self.plausible_relations: Dict[
            Tuple[SpanType, SpanType], List[Tuple[FrameType, SlotType]]
        ] = {}
        labels: Set[Any] = {"Nothing"}
        # For every span type that triggers a frame
        for frame_type in spec.frame_types:
            anchor_slot_type = self._frame_type_anchor(frame_type)
            for anchor_span_type in anchor_slot_type.types:
                if not isinstance(anchor_span_type, SpanType):
                    continue
                for slot_type in frame_type.slot_types:
                    if slot_type == anchor_slot_type:
                        continue
                    for filler_type in slot_type.types:
                        if not isinstance(filler_type, SpanType):
                            continue
                        self.plausible_relations.setdefault(
                            (anchor_span_type, filler_type), []
                        ).append((frame_type, slot_type))
                        labels.add((frame_type, slot_type))
        self.labels = list(labels)
        self.logger.debug(
            "plausible relations for slot classifier: " + str(self.plausible_relations)
        )

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        x, y = self.get_features_and_labels(corpus)

        if dev_corpus is None:
            self.cls = LinearSVC()
            self.cls.fit(x, y)
        else:
            best_f1 = -1.0
            best_c = 0.0
            best_cls = None
            for c_param in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100]:
                self.cls = LinearSVC(C=c_param, class_weight="balanced")
                self.cls.fit(x, y)
                self.logger.info("current c: " + str(c_param))
                micro_f1 = self.evaluate(dev_corpus)
                if micro_f1 > best_f1:
                    best_c = c_param
                    best_cls = copy.deepcopy(self.cls)
                    best_f1 = micro_f1
            assert best_cls is not None
            self.logger.info("Best c: " + str(best_c))
            self.cls = best_cls

    def predict(self, corpus: Corpus) -> None:
        x, _ = self.get_features_and_labels(corpus)
        y_pred = self.cls.predict(x)

    def evaluate(self, corpus: Corpus) -> float:

        """This function evaluates only the slot classifier, assuming
        the correct spans in gold"""

        x, y_gold = self.get_features_and_labels(corpus)
        y_pred = self.cls.predict(x)
        prec, reca, f1, supp = precision_recall_fscore_support(
            y_gold,
            y_pred,
            labels=[i for i, label in enumerate(self.labels) if label != "Nothing"],
            average="macro",
        )
        assert isinstance(f1, float)
        accuracy = accuracy_score(y_gold, y_pred)
        for score, name in [
            (prec, "Precision"),
            (reca, "Recall"),
            (f1, "F1-score"),
            (accuracy, "Accuracy"),
        ]:
            self.logger.info(name + "\t" + str(score))

        self.logger.debug("labels:")
        for i, label in enumerate(self.labels):
            self.logger.debug("%s\t%s", i, label)

        self.logger.debug("Confusion matrix:")
        self.logger.debug(confusion_matrix(y_gold, y_pred))

        return f1

    # Heuristic -- the first slot of every frame type is its anchor
    def _frame_type_anchor(self, frame_type: FrameType) -> SlotType:
        return frame_type.slot_types[0]

    def get_features_and_labels(
        self, corpus: Corpus
    ) -> Tuple[scipy.sparse.spmatrix, np.ndarray]:
        labels: List[Any] = []
        span1_list: List[Span] = []
        span2_list: List[Span] = []
        doc_list: List[Doc] = []
        graph_list: List[nx.Graph] = []
        idx2word_list: List[Dict[int, str]] = []
        prev_status = -1
        for i, instance in enumerate(corpus.instances):
            doc, graph, idx2word = self.preprocess_text(instance.text)
            span_pairs = self.get_span_pairs(instance)
            for span1, span2, relation in span_pairs:
                span1_list.append(span1)
                span2_list.append(span2)
                doc_list.append(doc)
                graph_list.append(graph)
                idx2word_list.append(idx2word)
                labels.append(relation)
            status = int(100 * i / len(corpus.instances))
            if prev_status != status and status % 10 == 0:
                self.logger.info("preprocessing %f%%" % (status))
            prev_status = status

        self.logger.info("preprocessing %f%%" % (100))
        # TOmaybeDO:
        # possibly give a list of all words in this
        # (list(iwl.values() for iwl in idx2word_list))
        self.logger.info("fitting count vectorizer")
        self.fit_count_vectorizers(
            span1_list, span2_list, doc_list, graph_list, idx2word_list
        )

        self.logger.info("creating features for train set")
        x = self.features_from_instance(
            span1_list, span2_list, doc_list, graph_list, idx2word_list
        )

        self.logger.debug("labels: " + str(self.labels))

        y = np.array([self.labels.index(label) for label in labels])

        bincount_y = np.bincount(y)
        self.logger.debug("counts: " + str(bincount_y))
        return x, y

    def get_span_pairs(self, instance: Instance) -> List[Tuple[Span, Span, Any]]:
        arcs: Dict[Tuple[Span, Span], Tuple[FrameType, SlotType]] = {}
        for frame in instance.frames:
            anchor_type = self._frame_type_anchor(frame.frame_type)
            anchor_slot = frame.slots[anchor_type]
            anchor_span = anchor_slot.fillers[0]
            if not isinstance(anchor_span, Span):
                continue
            for slot in frame.slots.values():
                for filler in slot.fillers:
                    if isinstance(filler, Frame):
                        continue
                    arcs[anchor_span, filler] = (frame.frame_type, slot.slot_type)
        span_pairs = []
        for x, y in product(instance.spans, instance.spans):
            if (x.span_type, y.span_type) in self.plausible_relations:
                arc = arcs.get((x, y))
                if arc is not None:
                    self.logger.debug("found relation: " + str(arc))
                    self.logger.debug(
                        "x.span_type: "
                        + str(x.span_type)
                        + ", y.span_type: "
                        + str(y.span_type)
                    )
                    self.logger.debug(
                        self.plausible_relations[x.span_type, y.span_type]
                    )
                span_pairs.append(
                    (
                        x,
                        y,
                        arc
                        if arc in self.plausible_relations[x.span_type, y.span_type]
                        else "Nothing",
                    )
                )
        return span_pairs

    def preprocess_text(self, text: str) -> Tuple[Doc, nx.Graph, Dict[int, str]]:
        doc = nlp(text)
        edges = []
        idx2word = {}
        # using the tokens themselves would require different instances of the
        # same token to compare? hash? to the same value, but different tokens
        # in the same sentence, that happen to have the same term need to hash?
        # compare? to different values. This is a more complicated assumption/
        # requirement than just looking at their offset, which should be
        # guaranteed to be unique (per sentence), and identical for multiple
        # instances of the same token.
        for token in doc:
            edges.append((token.idx, token.head.idx))
            idx2word[token.idx] = token.text
        graph = nx.Graph(edges)
        # print(edges)
        # print(graph)
        # print("---")
        return doc, graph, idx2word

    def fit_count_vectorizers(
        self,
        span1_list: List[Span],
        span2_list: List[Span],
        doc_list: List[Doc],
        graph_list: List[nx.Graph],
        idx2word_list: List[Dict[int, str]],
    ) -> None:

        _, shortest_path_list = self.get_shortest_path_features(
            doc_list, graph_list, span1_list, span2_list, idx2word_list
        )

        self.cv_text = CountVectorizer()
        span_text_list = [sp.text for sp in span1_list + span2_list]
        # self.logger.info("fitting word count vectorizer with:")
        # self.logger.info(span_text_list + shortest_path_list)
        self.cv_text.fit(span_text_list + shortest_path_list)

        self.cv_labels = CountVectorizer()
        span_label_list = [sp.span_type.name for sp in span1_list + span2_list]
        # self.logger.debug("fitting label count vectorizer with:")
        # self.logger.debug(span_label_list)
        self.cv_labels.fit(span_label_list)

    def get_shortest_path_features(
        self,
        doc_list: List[Doc],
        graph_list: List[nx.Graph],
        span1_list: List[Span],
        span2_list: List[Span],
        idx2word_list: List[Dict[int, str]],
    ) -> Tuple[np.ndarray, List[str]]:
        sp1_words = [
            self.find_node(doc, span1) for doc, span1 in zip(doc_list, span1_list)
        ]
        sp2_words = [
            self.find_node(doc, span2) for doc, span2 in zip(doc_list, span2_list)
        ]

        f_shortest_path = np.array(
            [
                self.edge_distance(graph, sp1, sp2)
                for graph, sp1, sp2 in zip(graph_list, sp1_words, sp2_words)
            ]
        )

        shortest_path_list = []
        for graph, sp1, sp2, idx2word in zip(
            graph_list, sp1_words, sp2_words, idx2word_list
        ):
            shortest_path = " ".join(
                [word for word in self.edge_words(graph, sp1, sp2, idx2word)]
            )
            shortest_path_list.append(shortest_path)

        return f_shortest_path, shortest_path_list

    def features_from_instance(
        self,
        span1_list: List[Span],
        span2_list: List[Span],
        doc_list: List[Doc],
        graph_list: List[nx.Graph],
        idx2word_list: List[Dict[int, str]],
    ) -> scipy.sparse.spmatrix:
        # spacy words

        f_sp1_text = self.cv_text.transform([span1.text for span1 in span1_list])
        f_sp2_text = self.cv_text.transform([span2.text for span2 in span2_list])

        f_sp1_label = self.cv_labels.transform(
            [span1.span_type.name for span1 in span1_list]
        )
        f_sp2_label = self.cv_labels.transform(
            [span2.span_type.name for span2 in span2_list]
        )

        f_shortest_path, shortest_path_list = self.get_shortest_path_features(
            doc_list, graph_list, span1_list, span2_list, idx2word_list
        )

        f_shortest_path = f_shortest_path.reshape(f_shortest_path.shape[0], 1)
        f_shortest_path_words = self.cv_text.transform(shortest_path_list)

        f_sequence_distance = np.array(
            [
                abs(self.find_node(doc, span1)[0].i - self.find_node(doc, span2)[0].i)
                for span1, span2, doc in zip(span1_list, span2_list, doc_list)
            ]
        )
        # np.ma: masked array; such that we only do logarithm on non-zero cells
        f_sequence_distance = np.ma.log(
            f_sequence_distance.reshape(f_sequence_distance.shape[0], 1)
        )

        self.logger.debug(f_sp1_text.shape)
        self.logger.debug(f_sp2_text.shape)
        self.logger.debug(f_sp1_label.shape)
        self.logger.debug(f_sp2_label.shape)
        self.logger.debug(f_shortest_path.shape)
        self.logger.debug(f_sequence_distance.shape)
        self.logger.debug(f_shortest_path_words.shape)

        X_feats = hstack(
            [
                f_sp1_text,
                f_sp2_text,
                f_sp1_label,
                f_sp2_label,
                csr_matrix(f_shortest_path),
                csr_matrix(f_sequence_distance),
                f_shortest_path_words,
            ]
        )

        return X_feats

    def save_model(self, filename: str) -> None:
        joblib.dump([self.cls, self.cv_text, self.cv_labels, self.labels], filename)

    def load_model(self, filename: str) -> None:
        self.cls, self.cv_text, self.cv_labels, self.labels = joblib.load(filename)

    @staticmethod
    def find_node(doc: Doc, span: Span) -> List[spacy.tokens.Token]:
        tokens = []
        for token in doc:
            token_left = token.idx
            token_right = token_left + len(token.text)
            # if the token starts within the span
            if span.left <= token_left < span.right:
                tokens.append(token)
            # if the token ends within the span
            elif span.left < token_right <= span.right:
                tokens.append(token)
            # if the token starts before the span and ends after the span
            elif token_left < span.left and token_right > span.right:
                tokens.append(token)
            # if the token starts after the end of the span, break
            elif token_left > span.right:
                break
        return tokens

    @staticmethod
    def edge_distance(
        graph: nx.Graph,
        tokens1: List[spacy.tokens.Token],
        tokens2: List[spacy.tokens.Token],
    ) -> int:
        try:
            shortest_length = min(
                nx.shortest_path_length(graph, token1.idx, token2.idx)
                for token1 in tokens1
                for token2 in tokens2
            )
            assert isinstance(shortest_length, int)
            return shortest_length
        except nx.NetworkXNoPath:
            return -1  # 10000 # float("inf")

    @staticmethod
    def edge_words(
        graph: nx.Graph,
        tokens1: List[spacy.tokens.Token],
        tokens2: List[spacy.tokens.Token],
        idx2word: Dict[int, str],
    ) -> List[str]:
        shortest = None
        for token1 in tokens1:
            for token2 in tokens2:
                try:
                    this = nx.shortest_path(graph, token1.idx, token2.idx)
                    if shortest is None or len(this) < len(shortest):
                        shortest = this
                except nx.NetworkXNoPath:
                    continue
        if shortest is None:
            return []
        return [idx2word[idx] for idx in shortest]
