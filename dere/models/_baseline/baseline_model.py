from __future__ import annotations

from typing import Optional, Dict, Any, IO

from dere.corpus import Corpus
from dere.models import Model
from dere.taskspec import TaskSpecification
from dere import Result

import numpy as np

from .span_classifier import SpanClassifier
from .slot_classifier import SlotClassifier


class BaselineModel(Model):
    def __init__(self, task_spec: TaskSpecification, model_spec: Dict[str, Any]) -> None:
        super().__init__(task_spec, model_spec)
        self._span_classifier = SpanClassifier(task_spec, model_spec.get('span_classifier', {}))
        self._slot_classifier = SlotClassifier(task_spec, model_spec.get('slot_classifier', {}))

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        self._span_classifier.train(corpus, dev_corpus=dev_corpus)
        self._slot_classifier.train(corpus, dev_corpus=dev_corpus)

    def predict(self, corpus: Corpus) -> None:
        self._span_classifier.predict(corpus)
        self._slot_classifier.predict(corpus)

    def initialize(self) -> None:
        self._span_classifier.initialize()
        self._slot_classifier.initialize()

    def dump(self, f: IO[bytes]) -> None:
        self._span_classifier.dump(f)
        self._slot_classifier.dump(f)

    def load(self, f: IO[bytes]) -> None:
        self._span_classifier.load(f)
        self._slot_classifier.load(f)

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...
