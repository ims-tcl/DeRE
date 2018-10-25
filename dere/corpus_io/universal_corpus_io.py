from typing import List, Type

from dere.corpus import Corpus

from dere.corpus_io import CorpusIO, BRATCorpusIO, CQSACorpusIO
from dere.taskspec import TaskSpecification


class UnknownCorpusFormatException(BaseException):
    ...


class UniversalCorpusIO(CorpusIO):
    def __init__(
        self,
        task_spec: TaskSpecification,
        corpus_io_classes: List[Type[CorpusIO]] = [BRATCorpusIO, CQSACorpusIO]
    ) -> None:
        super().__init__(task_spec)
        self._corpus_ios = [cio_class(task_spec) for cio_class in corpus_io_classes]

    def load(self, path: str, load_gold: bool = True) -> Corpus:
        for cio in self._corpus_ios:
            try:
                corpus = cio.load(path, load_gold)
                # if the corpus has no instances, we probably failed to load what we want
                if len(corpus.instances) == 0:
                    continue
                # Put this CorpusIO at the front of our list, so we default to it in the future.
                # Since we return immediately, it doesn't matter we are changing the list in the loop.
                self._corpus_ios.remove(cio)
                self._corpus_ios = [cio] + self._corpus_ios
                return corpus
            except Exception:
                continue
        raise UnknownCorpusFormatException

    def dump(self, corpus: Corpus, path: str, just_predictions: bool = True) -> None:
        for cio in self._corpus_ios:
            try:
                cio.dump(corpus, path, just_predictions)
                return
            except Exception:
                continue
        raise UnknownCorpusFormatException
