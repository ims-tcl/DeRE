# Author: Laura
from __future__ import annotations

import copy
import logging
import random
import pickle

from itertools import chain, combinations, product
from operator import mul
from typing import Optional, Dict, Tuple, List, Set, Any, Union, cast, Sequence, IO
from mypy_extensions import TypedDict

import networkx as nx
import numpy as np
import spacy
import spacy.cli

from spacy.tokens import Doc
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.externals import joblib
from scipy.sparse import hstack, csr_matrix, spmatrix, vstack
from sklearn.utils import shuffle

from dere.corpus import Corpus, Instance, Frame, Span, Slot, Filler
from dere.taskspec import TaskSpecification, FrameType, SpanType, SlotType
from dere.models import Model
from dere.utils import progressify

try:
    nlp = spacy.load("en")
except OSError:
    # This feels dirty :(
    # A better approach would be to depend upon the model as a module and let dependency handle it.
    # Unfortunately, those modules are not on PyPi, and so installation would be more complicated
    spacy.cli.download("en")
    nlp = spacy.load("en")

SpanPair = Tuple[Span, Span]
Edge = Tuple[FrameType, SlotType]
Label = Union[str, Edge]
Relation = Tuple[SpanPair, Label]
Arc = Tuple[FrameType, SlotType]
_ArrayLike = Union[List, np.ndarray, spmatrix]


class SlotClassifier(Model):
    def __init__(
            self, task_spec: TaskSpecification, model_spec: Dict[str, Any],
            seed: int = 98765
    ) -> None:
        super().__init__(task_spec, model_spec)
        self.seed = seed

        # TODO(Sean) move this to model.__init__
        self.logger = logging.getLogger("dere")

        # Find our plausible relations from the spec
        self.plausible_relations: Dict[Tuple[SpanType, SpanType], List[Edge]] = {}
        labels: Set[Any] = {"Nothing"}
        # For every span type that triggers a frame
        for frame_type in task_spec.frame_types:
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
            "[SlotClassifier] plausible relations for slot classifier: " + str(self.plausible_relations)
        )

    def initialize(self) -> None:
        random.seed(self.seed)
        self.cls: Optional[LinearSVC] = None
        self.cv_text: Optional[CountVectorizer] = None
        self.cv_labels: Optional[CountVectorizer] = None
        self.cv_deps_words: Optional[CountVectorizer] = None
        self.cv_sequence_text: Optional[CountVectorizer] = None

    def shuffle(self, X: _ArrayLike, y: _ArrayLike) -> Tuple[_ArrayLike, _ArrayLike]:
        X_shuffled, y_shuffled = shuffle(X, y, random_state=1111)
        return X_shuffled, y_shuffled

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        x_tmp, y_tmp, _ = self.get_features_and_labels(corpus, is_train=True)

        x, y = self.shuffle(x_tmp, y_tmp)

        self.logger.info("[SlotClassifier] Using " + str(x.shape[0]) + " instances for training")

        if dev_corpus is None:
            self.cls = LinearSVC(max_iter=10000)
            self.cls.fit(x, y)
        else:
            x_dev, y_dev, _ = self.get_features_and_labels(dev_corpus)
            best_f1 = -1.0
            best_c = 0.0
            best_cls = None
            self.logger.info("[SlotClassifier] Starting grid search...")
            for c_param in progressify([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100]):
                self.cls = LinearSVC(C=c_param, class_weight="balanced", max_iter=10000)
                self.cls.fit(x, y)
                self.logger.debug("[SlotClassifier] current c: " + str(c_param))
                micro_f1 = self._eval(dev_corpus, x=x_dev, y_gold=y_dev)
                if micro_f1 > best_f1:
                    best_c = c_param
                    best_cls = copy.deepcopy(self.cls)
                    best_f1 = micro_f1
            assert best_cls is not None
            self.logger.info("[SlotClassifier] Grid search done")
            self.logger.debug("[SlotClassifier] Best c: " + str(best_c))
            self.logger.info("[SlotClassifier] Training on all training data...")
            dev_x, dev_y, _ = self.get_features_and_labels(dev_corpus, is_train=False)
            assert(isinstance(x, spmatrix))
            train_x_all_tmp = vstack([x, dev_x])
            self.logger.debug("[SlotClassifier] train_x_all shape: " + str(train_x_all_tmp.shape))
            train_y_all_tmp = np.concatenate([y, dev_y])
            train_x_all, train_y_all = self.shuffle(train_x_all_tmp, train_y_all_tmp)
            self.cls = LinearSVC(C=best_c, class_weight="balanced", max_iter=10000)
            self.cls.fit(train_x_all, train_y_all)
            self.logger.info("[SlotClassifier] Training done")

    def predict(self, corpus: Corpus) -> None:
        assert self.cls is not None
        x, _, span_pairs = self.get_features_and_labels(corpus)
        if x.shape[0] == 0:
            # edge case -- we have no span pairs to classify
            # so there's nothing for us to do
            return
        self.logger.info("[SlotClassifier] Predicting relations")
        y_pred = self.cls.predict(x)
        predicted_labels = [self.labels[pi] for pi in y_pred]
        results_by_instance: List[List[Relation]] = []
        self.logger.info("[SlotClassifier] Generating frames")
        for (anchor_span, filler_span), predicted_label in zip(
            span_pairs, predicted_labels
        ):
            instance = anchor_span.instance
            for instance_results in results_by_instance:
                if instance_results[0][0][0].instance == instance:
                    instance_results.append(
                        ((anchor_span, filler_span), predicted_label)
                    )
                    break
            else:
                results_by_instance.append(
                    [((anchor_span, filler_span), predicted_label)]
                )
        for instance_results in results_by_instance:
            instance_results = self.filter_results(instance_results)
            self.generate_frames(instance_results)
        self.logger.info("[SlotClassifier] Finished generating frames")

    def filter_results(self, results: List[Relation]) -> List[Relation]:
        def filt(relation: Relation) -> bool:
            (anchor, filler), label = relation
            if label == "Nothing":
                return True
            type_pair = (anchor.span_type, filler.span_type)
            return label in self.plausible_relations[type_pair]

        return list(filter(filt, results))

    def generate_frames(self, results: List[Relation]) -> None:
        instance = results[0][0][0].instance
        anchored_frames: Dict[Span, Frame] = {}
        for (anchor, filler), label in results:
            if label == "Nothing":
                continue
            assert not isinstance(label, str)
            frame_type, slot_type = label
            if anchor in anchored_frames:
                frame = anchored_frames[anchor]
            else:
                frame = instance.new_frame(frame_type)
                anchor_slot = self.get_anchor_slot(frame)
                anchor_slot.add(anchor)
                anchored_frames[anchor] = frame
            frame.slots[slot_type].add(filler)
        self.split_overfilled_frames(instance)

    def split_overfilled_frames(self, instance: Instance) -> None:
        old_frames = list(instance.frames)
        for frame in old_frames:
            frame.remove()
            frame_type = frame.frame_type
            # Each element of prod corresponds to a particular slot
            # For each slot, we have a list of ways to fill that slot
            # Each way to fill that slot is a list of (SlotType, Filler) pairs
            prod: List[List[List[Tuple[SlotType, Filler]]]] = []
            for slot_type, slot in frame.slots.items():
                if slot_type.max_cardinality is None:
                    prod.append([[(slot_type, filler) for filler in slot.fillers]])
                else:
                    n = min(slot_type.max_cardinality, len(slot.fillers))
                    prod.append(
                        [
                            [(slot_type, ci) for ci in c]
                            for c in combinations(slot.fillers, n)
                        ]
                    )
            for assignment in product(*prod):
                new_frame = instance.new_frame(frame.frame_type)
                for term in assignment:
                    for slot_type, filler in term:
                        new_frame.slots[slot_type].add(filler)
                # check frame for min cardinality
                for slot_type, slot in new_frame.slots.items():
                    if slot_type.min_cardinality is not None:
                        if len(slot.fillers) < slot_type.min_cardinality:
                            new_frame.remove()
                            break

    def _eval(
            self,
            corpus: Corpus, x: Optional[spmatrix] = None,
            y_gold: Optional[np.ndarray] = None
    ) -> float:
        """
        This function evaluates only the slot classifier, assuming
        the correct spans in gold
        """
        assert self.cls is not None

        if x is None:
            x, y_gold, _ = self.get_features_and_labels(corpus)

        assert isinstance(x, spmatrix)
        assert isinstance(y_gold, np.ndarray)

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
            self.logger.info("[SlotClassifier] %r", name + "\t" + str(score))

        self.logger.debug("[SlotClassifier] labels:")
        for i, label in enumerate(self.labels):
            self.logger.debug("[SlotClassifier] %s\t%s", i, label)

        self.logger.debug("[SlotClassifier] Confusion matrix:")
        self.logger.debug("[SlotClassifier] %r", confusion_matrix(y_gold, y_pred))

        return f1

    # Heuristic -- the first slot of every frame type is its anchor
    def _frame_type_anchor(self, frame_type: FrameType) -> SlotType:
        return frame_type.slot_types[0]

    def get_anchor_slot(self, frame: Frame) -> Slot:
        return frame.slots[self._frame_type_anchor(frame.frame_type)]

    def get_features_and_labels(
        self, corpus: Corpus, is_train: bool = False
    ) -> Tuple[spmatrix, np.ndarray, List[SpanPair]]:

        labels: List[Any] = []
        span1_list: List[Span] = []
        span2_list: List[Span] = []
        doc_list: List[Doc] = []
        graph_list: List[nx.Graph] = []
        idx2word_list: List[Dict[int, str]] = []
        edge2dep_list = []
        sequence_words_list: List[Any] = []
        self.logger.info("[SlotClassifier] Getting features/labels...")
        for i, instance in enumerate(progressify(corpus.instances, "getting features")):
            instance_text = instance.text.replace('"', "'")
            doc, graph, idx2word, edge2dep = self.preprocess_text(instance_text)
            relations = self.get_relations(instance)
            for (span1, span2), relation in relations:
                span1_list.append(span1)
                span2_list.append(span2)
                doc_list.append(doc)
                graph_list.append(graph)
                idx2word_list.append(idx2word)
                edge2dep_list.append(edge2dep)
                sequence_words_list.append(self.build_sequence(doc, span1, span2))
                labels.append(relation)

        self.logger.info("[SlotClassifier] Getting features/labels done")
        # TOmaybeDO:
        # possibly give a list of all words in this
        # (list(iwl.values() for iwl in idx2word_list))
        self.logger.info("[SlotClassifier] Fitting count vectorizer...")
        # this is only for training
        if is_train:
            self.fit_count_vectorizers(
                span1_list,
                span2_list,
                doc_list,
                graph_list,
                idx2word_list,
                edge2dep_list,
                sequence_words_list,
            )
        self.logger.info("[SlotClassifier] Fitting count vectorizer done")

        set_name = "train" if is_train else "dev"
        self.logger.info("[SlotClassifier] Creating features for " + str(set_name) + " set...")
        x = self.features_from_instance(
            span1_list,
            span2_list,
            doc_list,
            graph_list,
            idx2word_list,
            edge2dep_list,
            sequence_words_list,
        )
        self.logger.info("[SlotClassifier] Creating features for " + str(set_name) + " set done")

        y = np.array([self.labels.index(label) for label in labels])

        if is_train:
            self.logger.debug("[SlotClassifier] labels: " + str(self.labels))
            bincount_y = np.bincount(y)
            self.logger.debug("[SlotClassifier] %r", str(bincount_y))

        return x, y, list(zip(span1_list, span2_list))

    def get_relations(self, instance: Instance) -> List[Relation]:
        arcs: Dict[SpanPair, Arc] = {}
        for frame in instance.frames:
            anchor_slot = self.get_anchor_slot(frame)
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
                arc = cast(Label, arcs.get((x, y)))
                if arc is not None:
                    self.logger.debug("[SlotClassifier] found relation: " + str(arc))
                    self.logger.debug(
                        "[SlotClassifier] x.span_type: "
                        + str(x.span_type)
                        + ", y.span_type: "
                        + str(y.span_type)
                    )
                    self.logger.debug(
                        "[SlotClassifier] %r",
                        self.plausible_relations[x.span_type, y.span_type],
                    )
                span_pairs.append(
                    (
                        (x, y),
                        arc
                        if arc in self.plausible_relations[x.span_type, y.span_type]
                        else "Nothing",
                    )
                )
        return span_pairs

    def preprocess_text(
        self, text: str
    ) -> Tuple[Doc, nx.Graph, Dict[int, str], Dict[Tuple[int, int], str]]:
        doc = nlp(text)
        edges = []
        idx2word = {}
        edge2dep = {}
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
            edge2dep[(token.idx, token.head.idx)] = token.dep_
            edge2dep[(token.head.idx, token.idx)] = token.dep_
        graph = nx.Graph(edges)
        self.logger.debug(
            "[SlotClassifier] preprocess text - edges: %r, graph: %r",
            edges,
            graph,
        )
        return doc, graph, idx2word, edge2dep

    def fit_count_vectorizers(
        self,
        span1_list: List[Span],
        span2_list: List[Span],
        doc_list: List[Doc],
        graph_list: List[nx.Graph],
        idx2word_list: List[Dict[int, str]],
        edge2dep_list: List[Dict[Tuple[int, int], str]],
        sequence_words_list: List[Any],
    ) -> None:

        _, shortest_path_list, path_list = self.get_shortest_path_features(
            doc_list,
            graph_list,
            span1_list,
            span2_list,
            idx2word_list,
            edge2dep_list,
        )

        self.logger.debug("[SlotClassifier] Fitting count vectorizer (text)...")
        self.cv_text = CountVectorizer()
        span_text_list = [sp.text for sp in span1_list + span2_list]
        self.cv_text.fit(span_text_list + shortest_path_list)

        self.logger.debug("[SlotClassifier] Fitting count vectorizer (labels)...")
        self.cv_labels = CountVectorizer()
        span_label_list = [sp.span_type.name for sp in span1_list + span2_list]
        self.cv_labels.fit(span_label_list)

        self.logger.debug("[SlotClassifier] Fitting count vectorizer (dep_words)...")
        self.cv_deps_words = CountVectorizer(ngram_range=(2, 2))  # type: ignore
        self.cv_deps_words.fit(path_list)

        self.logger.debug("[SlotClassifier] Fitting count vectorizer (sequence_text)...")
        self.cv_sequence_text = CountVectorizer()
        self.cv_sequence_text.fit(sequence_words_list)

    def get_shortest_path_features(
        self,
        doc_list: List[Doc],
        graph_list: List[nx.Graph],
        span1_list: List[Span],
        span2_list: List[Span],
        idx2word_list: List[Dict[int, str]],
        edge2dep_list: List[Dict[Tuple[int, int], str]],
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
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

        f_path_deps = np.array(
            [
                self.edge_words_deps(
                    graph,
                    self.find_node(doc, sp1),
                    self.find_node(doc, sp2),
                    idx2word,
                    edge2dep,
                )
                for graph, sp1, sp2, idx2word, edge2dep, doc in zip(
                    graph_list,
                    span1_list,
                    span2_list,
                    idx2word_list,
                    edge2dep_list,
                    doc_list,
                )
            ]
        )

        return f_shortest_path, shortest_path_list, f_path_deps

    def features_from_instance(
        self,
        span1_list: List[Span],
        span2_list: List[Span],
        doc_list: List[Doc],
        graph_list: List[nx.Graph],
        idx2word_list: List[Dict[int, str]],
        edge2dep_list: List[Dict[Tuple[int, int], str]],
        sequence_words_list: List[str],
    ) -> spmatrix:
        assert self.cv_text is not None
        assert self.cv_labels is not None
        assert self.cv_deps_words is not None
        assert self.cv_sequence_text is not None
        # spacy words

        self.logger.debug("[SlotClassifier] Getting features (text)...")
        f_sp1_text = self.cv_text.transform([span1.text for span1 in span1_list])
        f_sp2_text = self.cv_text.transform([span2.text for span2 in span2_list])

        self.logger.debug("[SlotClassifier] Getting features (labels)...")
        f_sp1_label = self.cv_labels.transform(
            [span1.span_type.name for span1 in span1_list]
        )
        f_sp2_label = self.cv_labels.transform(
            [span2.span_type.name for span2 in span2_list]
        )

        self.logger.debug("[SlotClassifier] Getting features (shortest path)...")
        shortest_path_features = self.get_shortest_path_features(
            doc_list, graph_list, span1_list, span2_list, idx2word_list, edge2dep_list
        )

        f_shortest_path, shortest_path_list, f_path_deps = shortest_path_features

        f_shortest_path = f_shortest_path.reshape(f_shortest_path.shape[0], 1)
        f_shortest_path_words = self.cv_text.transform(shortest_path_list)

        f_path_deps = self.cv_deps_words.transform(f_path_deps)  # type: ignore

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
        f_sequence_words = self.cv_sequence_text.transform(sequence_words_list)

        self.logger.debug("[SlotClassifier] f_sp1_text")
        self.logger.debug("[SlotClassifier] %r", f_sp1_text.shape)
        self.logger.debug("[SlotClassifier] f_sp2_text")
        self.logger.debug("[SlotClassifier] %r", f_sp2_text.shape)
        self.logger.debug("[SlotClassifier] f_sp1_label")
        self.logger.debug("[SlotClassifier] %r", f_sp1_label.shape)
        self.logger.debug("[SlotClassifier] f_sp2_label")
        self.logger.debug("[SlotClassifier] %r", f_sp2_label.shape)
        self.logger.debug("[SlotClassifier] f_shortest_path_distance")
        self.logger.debug("[SlotClassifier] %r", f_shortest_path.shape)
        self.logger.debug("[SlotClassifier] f_sequence_distance")
        self.logger.debug("[SlotClassifier] %r", f_sequence_distance.shape)
        self.logger.debug("[SlotClassifier] f_sequence_words")
        self.logger.debug("[SlotClassifier] %r", f_sequence_words.shape)
        self.logger.debug("[SlotClassifier] f_shortest_path_words")
        self.logger.debug("[SlotClassifier] %r", f_shortest_path_words.shape)
        self.logger.debug("[SlotClassifier] f_path_deps")
        self.logger.debug("[SlotClassifier] %r", f_path_deps.shape)

        X_feats = hstack(
            [
                f_sp1_text,
                f_sp2_text,
                f_sp1_label,
                f_sp2_label,
                csr_matrix(f_shortest_path),
                csr_matrix(f_sequence_distance),
                f_sequence_words,
                f_shortest_path_words,
                csr_matrix(f_path_deps),
            ]
        )

        return X_feats

    def dump(self, f: IO[bytes]) -> None:
        self.logger.info("[SlotClassifier] Saving model")
        joblib.dump(
            (
                random.getstate(),
                self.cls,
                self.cv_text,
                self.cv_labels,
                self.labels,
                self.cv_deps_words,
                self.cv_sequence_text,
            ),
            f,
        )

    def load(self, f: IO[bytes]) -> None:
        self.logger.info("[SlotClassifier] Loading model")
        (
            random_state,
            self.cls,
            self.cv_text,
            self.cv_labels,
            self.labels,
            self.cv_deps_words,
            self.cv_sequence_text,
        ) = joblib.load(f)
        random.setstate(random_state)

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

    def edge_words_deps(
        self,
        graph: nx.Graph,
        tokens1: List[spacy.tokens.Token],
        tokens2: List[spacy.tokens.Token],
        idx2word: Dict[int, str],
        edge2dep: Dict[Tuple[int, int], str],
    ) -> str:
        shortest = self.get_shortest_path(graph, tokens1, tokens2)
        if shortest is None:
            return ""
        one = iter(shortest)
        two = iter(shortest)
        next(two)
        words_deps = [idx2word[shortest[0]]]
        for left, right in zip(one, two):
            words_deps.append(edge2dep[left, right])
            words_deps.append(idx2word[right])

        return " ".join(words_deps)

    def edge_words(
        self,
        graph: nx.Graph,
        tokens1: List[spacy.tokens.Token],
        tokens2: List[spacy.tokens.Token],
        idx2word: Dict[int, str],
    ) -> List[str]:
        shortest = self.get_shortest_path(graph, tokens1, tokens2)
        if shortest is None:
            return []
        return [idx2word[idx] for idx in shortest]

    @staticmethod
    def get_shortest_path(
        graph: nx.Graph,
        tokens1: List[spacy.tokens.Token],
        tokens2: List[spacy.tokens.Token],
    ) -> Optional[List[int]]:
        shortest = None
        for token1 in tokens1:
            for token2 in tokens2:
                try:
                    this = nx.shortest_path(graph, token1.idx, token2.idx)
                    if shortest is None or len(this) < len(shortest):
                        shortest = this
                except nx.NetworkXNoPath:
                    continue
        return shortest  # type: ignore

    def build_sequence(self, doc: Doc, span1: Span, span2: Span) -> str:
        tokens1 = self.find_node(doc, span1)
        tokens2 = self.find_node(doc, span2)

        max1 = max(token.i for token in tokens1)
        min1 = min(token.i for token in tokens1)

        max2 = max(token.i for token in tokens2)
        min2 = min(token.i for token in tokens2)

        if max1 + min1 > max2 + min2:
            # 1 is after 2
            m = max2 + 1
            n = min1 - 1
        else:
            # 2 is after 1
            m = max1 + 1
            n = min2 - 1

        tokens = doc[m:n]  # may be empty
        return " ".join(t.text for t in tokens)
