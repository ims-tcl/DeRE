from __future__ import annotations

from typing import Optional, Dict, Any, IO, Union, Type, cast
import pickle

from dere.corpus import Corpus
from dere.taskspec import TaskSpecification
from dere.evaluation import Result


class Model:
    def __init__(
            self, task_spec: TaskSpecification, model_spec: Dict[str, Any],
            *args: Any, **kwargs: Any
    ) -> None:
        self.task_spec = task_spec
        self.model_spec = model_spec

    def initialize(self) -> None:
        '''
        Subclasses should overload this and initialize all model parameters here.
        '''
        ...

    def dump(self, f: IO[bytes]) -> None:
        '''
        Serialize all model parameters to a file, such that they can later be retrieved by load.
        While a default implementation is provided, subclasses should implement their own versions of this.

        Args:
            f: The file to write to.
        '''
        d = dict(self.__dict__)
        del d['task_spec']
        del d['model_spec']
        pickle.dump(d, f)

    def load(self, f: IO[bytes]) -> None:
        '''
        Reload model paramaters from a file, that were previously serialized with dump.
        While a default implementation is provided, subclasses should implement their own versions of this.

        Args:
            f: The file to read from.
        '''
        self.__dict__.update(pickle.load(f))

    # only minimal logic here, things that all models (might) need

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        ...

    def predict(self, corpus: Corpus) -> None:
        ...

    def evaluate(self, corpus: Corpus, gold: Corpus) -> Result:
        ...  # here there might actually be a sensible default implementation
