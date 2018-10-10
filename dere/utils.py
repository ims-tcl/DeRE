from typing import Type, List, Any, Optional
import inspect

from dere.models import Model
from dere.corpus import Corpus


class GridSearchModel(Model):
    def __init__(
            self,
            base_class: Type[Model],
            param_name: str,
            param_values: List[Any],
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.base_class = base_class
        self.param_name = param_name
        self.param_values = param_values
        self.models: List[Model] = []
        for value in param_values:
            kwargs.update({param_name: value})
            self.models.append(base_class(*args, **kwargs))
        self.task_spec = self.models[0].task_spec

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        # TODO
        if dev_corpus is None:
            return
        best_model = None
        best_score = None
        for model, value in zip(self.models, self.param_values):
            print("Grid Search: %s: %s" % (self.param_name, str(value)))
            model.train(corpus, dev_corpus)
            score = model.evaluate(corpus, dev_corpus)
            if best_model is None or score < best_score:
                best_model = model
                best_score = score
                self.best_model = best_model

    def predict(self, corpus: Corpus) -> None:
        return self.best_model.predict(corpus)

    '''
    def evaluate(self, corpus: Corpus, gold: Corpus) -> Result:
        return self.best_model.evaluate(corpus, gold)
    '''


class grid_search:
    # a callable that takes a class and return a new class/factory
    def __init__(self, **kwargs: List[Any]) -> None:
        self.param_ranges = kwargs

    def __call__(self, decoratee_class):
        for param_name, param_values in self.param_ranges.items():
            decoratee_class = _GridSearchModelFactory(decoratee_class, param_name, param_values)
        return decoratee_class


class _GridSearchModelFactory:
    def __init__(self, base_class: Type[Model], param_name: str, param_values: List[str]) -> None:
        self.base_class = base_class
        self.param_name = param_name
        self.param_values = param_values

    def __call__(self, *args, **kwargs):
        # if they specify the parameter on instantiation, then we don't grid search over it
        if self.param_name in args:
            return self.base_class(*args, **kwargs)
        else:
            return GridSearchModel(self.base_class, self.param_name, self.param_values, *args, **kwargs)


"""
def _grid_search_class(
    decoratee_class: Type[Model],
        param_name: str,
        param_values: List[Any],
        param_names: List[str]
) -> Type[Model]:
    class GridSearchModel(Model):
        def __new__(cls, *args, **kwargs):
            # if a user is specifically specifying a hyperparameter value
            # then do not do a grid search over this, but instead return
            # the undecorated class
            if param_name in kwargs or param_name in param_names[:len(args)]:
                return decoratee_class(*args, **kwargs)
            else:
                return super().__new__(cls)

        def __init__(self, *args, **kwargs):
            self.models: List[Model] = []
            for value in param_values:
                kwargs.update({param_name: value})
                self.models.append(decoratee_class(*args, **kwargs))

        def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
            # TODO
            if dev_corpus is None:
                return
            best_model = None
            best_score = None
            for model, value in zip(self.models, param_values):
                print("Grid Search: %s: %s" % (param_name, str(value)))
                model.train(corpus, dev_corpus)
                score = model.evaluate(corpus, dev_corpus)
                if best_model is None or score < best_score:
                    best_model = model
                    best_score = score
                    self.best_model = best_model

        def predict(self, corpus: Corpus) -> None:
            return self.best_model.predict(corpus)

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


class dummy_decorator:
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return dummy_class(self.cls, *args, **kwargs)


class dummy_class:
    def __init__(self, cls, *args, **kwargs):
        self.decoratee = cls(*args, **kwargs)


"""
