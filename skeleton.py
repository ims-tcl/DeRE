import pickle
from typing import List

import click


class Corpus:
    def __init__(self):
        self.instances: List[Instance] = []


class Annotation:
    ...


class Result:
    def __init__(self, **metrics):
        for metric in metrics:
            setattr(self, metric, metrics[metric])

    def __str__(self) -> str:
        ...  # "toString"

    def __repr__(self) -> str:
        ...  # "print"

    def __sub__(self, other):  # Result) -> Result:
        ...  # "compare"
        # r1 = Result(...)
        # r2 = Result(...)
        # difference = r2 - r1


class CorpusReader:
    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = corpus_path

    def load(self) -> Corpus:
        ...


class BRATCorpusReader(CorpusReader):
    def load(self) -> Corpus:
        ...


class XML123CorpusReader(CorpusReader):
    def load(self) -> Corpus:
        ...


class Instance:
    def __init__(self, text: str, annotations: List[Annotation]) -> None:
        self.text = text
        self.annotations = []


class SpanAnnotation(Annotation):
    def __init__(self, left: int, right: int, type_: str) -> None:
        self.left = left
        self.right = right
        self.type_ = type_


class SlotAnnotation(Annotation):
    def __init__(self, type_: str) -> None:
        self.type_ = type_


class FrameAnnotation(Annotation):
    def __init__(self, slots: List[SlotAnnotation]) -> None:
        self.slots = slots


class FrameRepresentation:
    def __init__(self, schema_file: str) -> None:
        ...


class Model:
    def __init__(self, schema_file: str) -> None:
        self.fr = FrameRepresentation(schema_file)

    # only minimal logic here, things that all models (might) need
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    def eval(self, corpus: Corpus, predicted: list) -> Result:
        ...  # here there might actually be a sensible default implementation


class BaselineModel(Model):
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...


class PGModel(Model):
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    def eval(self, corpus: Corpus, predicted: list) -> Result:
        ...


CORPUS_READERS = {"BRAT": BRATCorpusReader, "XML123": XML123CorpusReader}

MODELS = {"baseline": BaselineModel, "pgm": PGModel}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model", default="baseline")
@click.option("--schema", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(model, schema, outfile):
    print("building with", model, schema, outfile)
    model = MODELS[model](schema)
    with open(outfile, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="bare_model.pkl")
@click.option("--outfile", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def train(corpus_path, model, outfile, corpus_format):
    print("training with", corpus_path, model, outfile)
    with open(model, "rb") as f:
        model = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    model.train(corpus)
    with open(outfile, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def predict(corpus_path, model, corpus_format):
    print("predicting with", corpus_path, model)
    with open(model, "rb") as f:
        model = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    predictions = model.predict(corpus)
    print(predictions)  # or something smarter


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def evaluate(corpus_path, model, corpus_format):
    print("evaluating with", corpus_path, model)
    with open(model, "rb") as f:
        model = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    predictions = model.predict(corpus)
    result = model.eval(corpus, predictions)
    print(result)  # or something smarter


if __name__ == "__main__":
    cli()
