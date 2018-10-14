from typing import Type, List, Any, Optional, Dict, IO
import inspect
import pickle

from dere.models import Model
from dere.corpus import Corpus
from dere.taskspec import TaskSpecification


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
