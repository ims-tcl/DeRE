import click
import pickle

# path hackery to get imports working as intended
import sys
import os
path = os.path.dirname(sys.modules[__name__].__file__)  # noqa
path = os.path.join(path, "..")  # noqa
sys.path.insert(0, path)  # noqa

from dere.readers import CorpusReader, BRATCorpusReader, XML123CorpusReader
from dere.models import BaselineModel, PGModel
from dere.schema import load_task_schema_file


CORPUS_READERS = {"BRAT": BRATCorpusReader, "XML123": XML123CorpusReader}

MODELS = {"baseline": BaselineModel, "pgm": PGModel}


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--model", default="baseline")
@click.option("--schema", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(model: str, schema: str, outfile: str) -> None:
    _build(model, schema, outfile)


def _build(model_name: str, schema_path: str, out_path: str) -> None:
    print("building with", model_name, schema_path, out_path)
    schema = load_task_schema_file(schema_path)
    model = MODELS[model_name](schema)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="bare_model.pkl")
@click.option("--outfile", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def train(
    corpus_path: str, model: str, outfile: str, corpus_format: str
) -> None:
    _train(corpus_path, model, outfile, corpus_format)


def _train(
    corpus_path: str, model_path: str, out_path: str, corpus_format: str
) -> None:
    print("training with", corpus_path, model_path, out_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    model.train(corpus)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def predict(corpus_path: str, model: str, corpus_format: str) -> None:
    _predict(corpus_path, model, corpus_format)


def _predict(
    corpus_path: str, model_path: str, corpus_format: str
) -> None:
    print("predicting with", corpus_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    predictions = model.predict(corpus)
    print(predictions)  # or something smarter


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def evaluate(corpus_path: str, model: str, corpus_format: str) -> None:
    _evaluate(corpus_path, model, corpus_format)


def _evaluate(
    corpus_path: str, model_path: str, corpus_format: str
) -> None:
    print("evaluating with", corpus_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    predictions = model.predict(corpus)
    result = model.eval(corpus, predictions)
    print(result)  # or something smarter


if __name__ == "__main__":
    cli()
