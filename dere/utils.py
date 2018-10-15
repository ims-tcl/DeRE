from typing import Sequence, Generator, TypeVar, Type, List, Any, Optional, Dict, IO
import inspect
import pickle
import sys

from dere.models import Model
from dere.corpus import Corpus
from dere.taskspec import TaskSpecification


T = TypeVar("T")


def progressify(seq: Sequence[T], string: str = "") -> Generator[T, None, None]:
    """
    Display a progress bar in the terminal while iterating over a sequence.

    This function can be used whereever we iterate over a sequence (i.e.
    something iterable with a length) and we want to display progress being
    made to the user. It works by passing on the values from the sequence (by
    yielding them), which printing an updated progress bar to the terminal. The
    progress is estimated based on how far we are through the sequence (based
    on our iteration step and its length). The progress bar is as long as its
    length, unless that is larger than the maximum length to avoid overflowing
    on small terminal sizes or large input. The maximum length is set to 30, to
    leave some space for a message, that can be printed along with the progress
    bar itself (even on old standard 80-column terminals).

    Printing the progress bar works through the use of block-drawing characters
    and carriage returns (ASCII code 0x0d). In order to avoid the cursor hiding
    part of the progress bar, we use terminal escape codes to temporarily hide
    it during the runtime of this generator.

    Args:
        seq: The sequence of elements to iterate over (often a list).
        string: An optional message that can be printed along with the bar. It
            can contain the special sequence "%i", which will be replaced with
            a string representation of the current item.

    Yields:
        The elements from seq.

    Example:
        for element in progressify([0.01, 0.1, 0.25, 0.5, 0.9, 0.99]):
            do_something_with(element)  # progress bar is updated at each step
    """
    try:
        print("\033[?25l")  # hide the cursor
        length = len(seq)
        maxlen = 30
        for i, element in enumerate(seq):
            if length > maxlen:
                i = int(i * maxlen/length)
            print(
                "\r[{}{}] {}".format(
                    "▓"*(i+1),
                    "░"*((length if length <= maxlen else maxlen)-i-1),
                    string.replace("%i", str(element)),
                ),
                file=sys.stderr,
                flush=True,  # to see the updated bar immediately
                end=""
            )
            yield element
    finally:
        print("\033[?25h")  # show the cursor again
        print()


def _grid_search_class(
    decoratee_class: Type[Model],
        param_name: str,
        param_values: List[Any],
        param_names: List[str]
) -> Type[Model]:
    class GridSearchModel(Model):
        def __new__(
                cls,
                task_spec: TaskSpecification, model_spec: Dict[str, Any],
                *args: Any, **kwargs: Any
        ) -> Model:
            # if a user is specifically specifying a hyperparameter value
            # then do not do a grid search over this, but instead return
            # the undecorated class
            if param_name in kwargs or param_name in param_names[:len(args)]:
                return decoratee_class(task_spec, model_spec, *args, **kwargs)
            else:
                return super().__new__(cls)  # type: ignore

        def __init__(
                self, task_spec: TaskSpecification, model_spec: Dict[str, Any],
                *args: Any, **kwargs: Any
        ) -> None:
            super().__init__(task_spec, model_spec)
            self.models: List[Model] = []
            for value in param_values:
                kwargs.update({param_name: value})
                self.models.append(decoratee_class(task_spec, model_spec, *args, **kwargs))

        def initialize(self) -> None:
            self.best_model: Optional[Model] = None
            self.best_model_index: Optional[int] = None
            for model in self.models:
                model.initialize()

        def dump(self, f: IO[bytes]) -> None:
            pickle.dump(self.best_model_index, f)
            for model in self.models:
                model.dump(f)

        def load(self, f: IO[bytes]) -> None:
            self.best_model_index = pickle.load(f)
            for model in self.models:
                model.load(f)
            self.best_model = None
            if self.best_model_index is not None:
                self.best_model = self.models[self.best_model_index]

        def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
            # TODO
            if dev_corpus is None:
                return
            best_model_index = None
            best_score = None
            for i, (model, value) in enumerate(zip(self.models, param_values)):
                print("Grid Search: %s: %s" % (param_name, str(value)))
                model.train(corpus, dev_corpus)
                score = model.evaluate(dev_corpus)
                if best_model_index is None or score > best_score:
                    print("Best score!")
                    best_model_index = i
                    best_score = score
                else:
                    print("No improvement")
                print(score.report())
            # keep mypy happy; thought I guess if we have zero models this would be applicable
            if best_model_index is not None:
                self.best_model_index = best_model_index
                self.best_model = self.models[best_model_index]

        def predict(self, corpus: Corpus) -> None:
            assert self.best_model is not None
            self.best_model.predict(corpus)

        '''
        def evaluate(self, corpus: Corpus, gold: Corpus) -> Result:
            return self.best_model.evaluate(corpus, gold)
        '''
    return GridSearchModel


def grid_search(**kwargs):
    def grid_search_decorator(model_class):
        param_names = list(inspect.signature(model_class).parameters.keys())
        for param in kwargs:
            model_class = _grid_search_class(model_class, param, kwargs[param], param_names)
        return model_class
    return grid_search_decorator
