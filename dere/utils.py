from typing import Type, List, Any, Optional
import inspect

from dere.models import Model
from dere.corpus import Corpus


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
