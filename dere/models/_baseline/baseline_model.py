from __future__ import annotations

from typing import Optional, Dict, Any
from mypy_extensions import TypedDict

from dere.corpus import Corpus
from dere.models import Model
from dere.taskspec import TaskSpecification
from dere import Result

import numpy as np

from .span_classifier import SpanClassifier
from .slot_classifier import SlotClassifier


class BaselineModel(Model):
    class ModelSpec(TypedDict, total=False):
        span_classifier: SpanClassifier.ModelSpec
        slot_classifier: SlotClassifier.ModelSpec

    def __init__(self, task_spec: TaskSpecification, model_spec: BaselineModel.ModelSpec) -> None:
        super().__init__(task_spec, model_spec)
        self._span_classifier = SpanClassifier(task_spec, model_spec.get('span_classifier', {}))
        self._slot_classifier = SlotClassifier(task_spec, model_spec.get('slot_classifier', {}))

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        self._span_classifier.train(corpus, dev_corpus=dev_corpus)
        self._slot_classifier.train(corpus, dev_corpus=dev_corpus)

    def predict(self, corpus: Corpus) -> None:
        self._span_classifier.predict(corpus)
        self._slot_classifier.predict(corpus)

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...
