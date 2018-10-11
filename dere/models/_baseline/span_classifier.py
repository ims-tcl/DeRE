# Author: Heike, Sean
from __future__ import annotations

import copy
import logging
import string
import random
import os
from typing import Dict, List, Tuple, Set, Optional, Union, Any, cast
from mypy_extensions import TypedDict

from mypy_extensions import TypedDict
from sklearn_crfsuite import CRF, metrics
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


from dere.taskspec import TaskSpecification, SpanType
from dere.corpus import Corpus, Instance, Span
from dere.models import Model

from sklearn.externals import joblib

word_tokenizer = TreebankWordTokenizer()

Features = Dict[str, Union[str, bool]]


class SpanClassifier(Model):
    def __init__(self, task_spec: TaskSpecification, model_spec: Dict[str, Any]) -> None:
        super().__init__(task_spec, model_spec)
        self.target_span_types = list(task_spec.span_types)
        self.gazetteer: Dict[str, Set[str]] = {}
        try:
            gazetteer_filename = model_spec['gazeteer']
            self.read_gazetteer(gazetteer_filename)
        except KeyError:
            pass
        self.logger = logging.getLogger(__name__)
        self.target2classifier: Dict[str, CRF] = {}
        self.ps = PorterStemmer()
        # @Sean: TODO: read from spec which spans are given (if any)

        given_span_types = [task_spec.type_lookup("Protein")]
        given_span_types = [gst for gst in given_span_types if gst is not None]
        for given_span_type in given_span_types:
            assert type(given_span_type) is SpanType
        self.given_span_types = cast(List[SpanType], given_span_types)
        for t in self.target_span_types:
            if t in self.given_span_types:  # no need to predict given spans
                self.target_span_types.remove(t)
        # initialize everything necessary

    def shuffle(self, X: List, y: List) -> Tuple[List, List]:
        indices = [i for i in range(len(X))]
        # using shuffle instead of permutation to reproduce set from old code
        # TODO: change to permutation as this is more elegant
        random.seed(1111)
        random.shuffle(indices)
        X_shuffled = [X[i] for i in indices]
        y_shuffled = [y[i] for i in indices]
        return X_shuffled, y_shuffled

    def train(
        self,
        corpus_train: Corpus,
        dev_corpus: Optional[Corpus] = None,
    ) -> None:
        self.logger.info("extracting features")
        X_train = self.get_features(corpus_train)

        self.logger.info("using " + str(len(X_train)) + " sentences for training")

        if dev_corpus is not None:
            X_dev = self.get_features(dev_corpus)

        self.logger.info(
            "target span types: " + str([st.name for st in self.target_span_types])
        )

        for t in self.target_span_types:
            X_train2 = self.get_span_type_specific_features(corpus_train, t)
            X_train_merged = X_train
            # X_train_merged = self.merge_features(X_train, X_train2)
            self.logger.info("Optimizing classifier for class " + str(t))
            target_t = self.get_binary_labels(corpus_train, t, use_bio=True)
            self.logger.debug(target_t)

            X_train_merged, target_t = self.shuffle(X_train_merged, target_t)

            if dev_corpus is None:
                aps = True
                c2v = 0.1
                default_setup = {"aps": aps, "c2v": c2v}
                self.logger.info(
                    "No dev corpus given. Using setup: " + str(default_setup)
                )
                crf = CRF(
                    algorithm="l2sgd",
                    all_possible_transitions=True,
                    all_possible_states=aps,
                    c2=c2v,
                )
                crf.fit(X_train_merged, target_t)
                self.target2classifier[t.name] = crf
            else:
                # get features for dev corpus
                X_dev2 = self.get_span_type_specific_features(dev_corpus, t)
                X_dev_merged = X_dev
                # X_dev_merged = self.merge_features(X_dev, X_dev2)
                y_dev = self.get_binary_labels(dev_corpus, t, use_bio=True)
                # optimize on dev
                best_f1 = -1.0
                Setup = TypedDict("Setup", {"aps": bool, "c2v": float})
                best_setup: Setup = {"c2v": 0.001, "aps": True}
                stopTraining = False
                for aps in [True, False]:
                    if stopTraining:
                        break
                    for c2v in [
                        0.001,
                        0.01,
                        0.1,
                        0.3,
                        0.5,
                        0.6,
                        0.9,
                        0.99,
                        1.0,
                        1.3,
                        1.6,
                        3.0,
                        6.0,
                        10.0,
                    ]:
                        if stopTraining:
                            break
                        cur_setup: Setup = {"aps": aps, "c2v": c2v}
                        self.logger.info("Current setup: " + str(cur_setup))
                        crf = CRF(
                            algorithm="l2sgd",
                            all_possible_transitions=True,
                            all_possible_states=aps,
                            c2=c2v,
                        )
                        crf.fit(X_train_merged, target_t)

                        micro_f1 = self._eval(crf, X_dev_merged, y_dev)
                        if micro_f1 > best_f1:
                            self.target2classifier[t.name] = crf
                            best_setup = cur_setup
                            best_f1 = micro_f1
                        if micro_f1 == 1.0:  # cannot get better
                            stopTraining = True
                self.logger.info("Best setup: " + str(best_setup))
                with open("best_parameters_span", 'a') as out:
                    out.write(t.name)
                    out.write("\n")
                    for k, v in best_setup.items():
                        out.write(str(k) + "\t" + str(v) + "\n")
                    out.write("\n")
                self.logger.info("Retraining best setup with all available data")
                X_train_all = X_train_merged + X_dev_merged
                target_all = target_t + y_dev
                X_train_all, target_all = self.shuffle(X_train_all, target_all)
                crf = CRF(
                        algorithm="l2sgd",
                        all_possible_transitions=True,
                        all_possible_states=best_setup["aps"],
                        c2=best_setup["c2v"]
                )
                crf.fit(X_train_all, target_all)
                self.target2classifier[t.name] = crf

    def _eval(
        self, classifier: CRF, X_dev: List[List[Features]], y_dev: List[List[str]]
    ) -> float:
        y_pred = classifier.predict(X_dev)
        try:
            self.logger.info(
                metrics.flat_classification_report(
                    y_dev, y_pred, labels=["I", "B"], digits=3
                )
            )
        except ZeroDivisionError:
            pass
        micro_f1 = metrics.flat_f1_score(
            y_dev, y_pred, average="micro", labels=["I", "B"]
        )
        self.logger.info("micro F1: " + str(micro_f1))
        return micro_f1

    def predict(self, corpus: Corpus) -> None:
        # input:
        # raw_data: list of list of Token instances

        if self.target_span_types is None:
            self.logger.error(
                "target span types are not initialized. Use train function"
                + " first or load existing model to initialize it"
            )
            return []
        X_test = self.get_features(corpus)
        predictions = {}
        for t in self.target_span_types:
            self.logger.debug(t)
            X_test2 = self.get_span_type_specific_features(corpus, t)
            X_test_merged = X_test
            # X_test_merged = self.merge_features(X_test, X_test2)
            y_pred = self.target2classifier[t.name].predict(X_test_merged)
            for X_item, y_item in zip(X_test_merged, y_pred):
                self.logger.debug(X_item)
                self.logger.debug(y_item)
            self.logger.debug("-----")
            predictions[t.name] = y_pred
        self.prepare_results(predictions, corpus)

    def get_spans_for_tokens(
        self, tokens: List[Tuple[int, int]], instance: Instance
    ) -> List[List[Span]]:
        spans_for_tokens = []  # one list of spans per token
        for (t_left, t_right) in tokens:
            token_spans = []
            for s in instance.spans:
                # because of multi-token annotations:
                if s.left <= t_left and t_right <= s.right:
                    token_spans.append(s)
                else:
                    """
                    if tokenization did not split a word in the same way as
                    annotation
                    Example: train id: 10229815: "CTLA-4-Mediated" -> is not
                    split into two tokens but annotation is only for first part
                    CTLA-4
                    solution here: adapt label but do not adapt tokenization
                    because it cannot be adapted for unseed test data either
                    two special cases:
                    """
                    if s.left == t_left and t_right > s.right:
                        token_spans.append(s)
                    elif t_right == s.right and t_left < s.left:
                        token_spans.append(s)
            spans_for_tokens.append(token_spans)
        # sanity check: we need one label for each token
        assert len(spans_for_tokens) == len(tokens)
        return spans_for_tokens

    def get_binary_labels(
        self, corpus: Corpus, t: SpanType, use_bio: bool = False
    ) -> List[List[str]]:
        binary_labels = []
        for instance in corpus.instances:
            instance_text = instance.text.replace('"', "'")
            instance.text = instance_text
            instance_tokens = list(word_tokenizer.span_tokenize(instance_text))
            spans_for_tokens = self.get_spans_for_tokens(instance_tokens, instance)

            # do binarization based on given target label t
            instance_binary_labels: List[str] = []

            for idx, token_spans in enumerate(spans_for_tokens):
                for span in token_spans:
                    if span.span_type == t:
                        if use_bio:
                            if idx > 0 and span in spans_for_tokens[idx - 1]:
                                instance_binary_labels.append("I")
                            else:
                                instance_binary_labels.append("B")
                        else:
                            instance_binary_labels.append("1")
                        break
                else:  # for-else: if we don't find a span of our type
                    if use_bio:
                        instance_binary_labels.append("O")
                    else:
                        instance_binary_labels.append("0")

            # sanity check: we need one label for each token
            assert len(instance_binary_labels) == len(instance_tokens)

            binary_labels.append(instance_binary_labels)
        return binary_labels

    def read_gazetteer(self, filename: str) -> None:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                parts = line.split(" ")
                label = parts[0]
                example = " ".join(parts[1:])
                if label not in self.gazetteer:
                    self.gazetteer[label] = set()
                self.gazetteer[label].add(example.lower())

    def get_span_type_specific_features(
        self, corpus: Corpus, target: SpanType
    ) -> List[List[Features]]:
        feature_list = []
        for instance in corpus.instances:
            instance_text = instance.text.replace('"', "'")
            instance.text = instance_text
            words = list(word_tokenizer.tokenize(instance_text))
            instance_feature_list = []
            for word in words:
                features: Features = {}
                if target.name in self.gazetteer:
                    features["in_" + str(target.name) + "_gazetteer"] = (
                        word.lower() in self.gazetteer[target.name]
                    )
                instance_feature_list.append(features)
            feature_list.append(instance_feature_list)
        return feature_list

    def merge_features(
        self, features1: List[List[Features]], features2: List[List[Features]]
    ) -> List[List[Features]]:
        result = copy.deepcopy(features1)
        for instance_features1, instance_features2 in zip(result, features2):
            for f1, f2 in zip(instance_features1, instance_features2):
                f1.update(f2)
        return result

    def word_features(self, word: str, prefix: str) -> Features:
        return {
            prefix + ".lower": word.lower(),
            prefix + ".isupper": word.isupper(),
            prefix + ".istitle": word.istitle(),
            prefix + ".isdigit": word.isdigit(),
            prefix + ".containsdigit": self.contains_digit(word),
            prefix + ".containspunct": self.contains_punct(word),
            prefix + ".stem": self.get_stem(word),
        }

    def token_features(
        self, instance: Instance, token: Tuple[int, int], prefix: str
    ) -> Features:
        left, right = token
        word = instance.text[left:right]
        features = self.word_features(word, prefix)
        for given_span_type in self.given_span_types:
            features[
                prefix + ".is_" + str(given_span_type.name)
            ] = self.is_token_in_span(instance, token, given_span_type)
        return features

    def get_features(self, corpus: Corpus) -> List[List[Features]]:
        feature_list = []
        for instance in corpus.instances:
            instance_text = instance.text.replace('"', "'")
            instance.text = instance_text
            token_spans = list(word_tokenizer.span_tokenize(instance_text))
            instance_feature_list = []
            for i, token in enumerate(token_spans):
                features = self.token_features(instance, token, "word")

                # context features: features of previous token
                if i > 0:
                    prev_token = token_spans[i - 1]
                    features.update(
                        self.token_features(instance, prev_token, "-1:word")
                    )
                else:
                    features["BOS"] = True

                # context features: features of next token
                if i < len(token_spans) - 1:
                    next_token = token_spans[i + 1]
                    features.update(
                        self.token_features(instance, next_token, "+1:word")
                    )
                else:
                    features["EOS"] = True

                # for key in features:
                #    if isinstance(features[key], bool):
                #        if features[key]:
                #            features[key] = "1"
                #        else:
                #            features[key] = "0"

                instance_feature_list.append(features)

            feature_list.append(instance_feature_list)
        return feature_list

    def is_token_in_span(
        self, instance: Instance, token: Tuple[int, int], span_type: SpanType
    ) -> bool:
        left, right = token
        for span in instance.spans:
            if span.span_type == span_type:
                if span.left <= left and span.right >= right:
                    return True
                else:  # If tokenizer does not split the same way as annotation
                    if span.left == left and right > span.right:
                        return True
                    elif span.right == right and left < span.left:
                        return True
        return False

    def contains_digit(self, word: str) -> bool:
        return len(set(word) & set(string.digits)) > 0

    def contains_punct(self, word: str) -> bool:
        # TODO -- change to string.punctuation
        punct = ".,-"
        return len(set(word) & set(punct)) > 0

    def get_stem(self, word: str) -> str:
        return self.ps.stem(word)

    def prepare_results(
        self, predictions: Dict[str, List[List[str]]], corpus: Corpus
    ) -> None:
        for i, instance in enumerate(corpus.instances):
            instance_text = instance.text.replace('"', "'")
            instance.text = instance_text
            instance_tokens = list(word_tokenizer.span_tokenize(instance_text))
            for target_span_type in self.target_span_types:
                instance_predictions = predictions[target_span_type.name][i]
                current_span_left: Optional[int] = None
                current_span_right = 0
                for token, label in zip(instance_tokens, instance_predictions):
                    if label != "O":
                        d_msg = "token: " + str(token) + "\t" + str(instance.text[token[0]:token[1]])
                        d_msg += "\t" + "label: " + str(target_span_type)
                        self.logger.debug(d_msg)
                    if current_span_left is not None and label in "BO":
                        instance.new_span(
                            target_span_type,
                            current_span_left,
                            current_span_right,
                        )
                        current_span_left = None
                    if label == "B":
                        current_span_left = token[0]
                    if label in "BI":
                        current_span_right = token[1]
                if current_span_left is not None:  # otherwise spans at end of window are neglected
                    instance.new_span(
                            target_span_type,
                            current_span_left,
                            current_span_right,
                    )
