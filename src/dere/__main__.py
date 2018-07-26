import click
import pickle
import logging
from typing import Optional

# path hackery to get imports working as intended
import sys
import os

path = os.path.dirname(sys.modules[__name__].__file__)  # noqa
path = os.path.join(path, "..")  # noqa
sys.path.insert(0, path)  # noqa

from dere.corpus_io import CorpusIO, BRATCorpusIO, CQSACorpusIO
from dere.models import BaselineModel, PGModel, NOPModel
import dere.taskspec


CORPUS_IOS = {"BRAT": BRATCorpusIO, "CQSA": CQSACorpusIO}

MODELS = {"baseline": BaselineModel, "pgm": PGModel, "nop": NOPModel}

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--model", default="baseline")
@click.option("--spec", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(model: str, spec: str, outfile: str) -> None:
    _build(model, spec, outfile)


def _build(model_name: str, spec_path: str, out_path: str) -> None:
    print("building with", model_name, spec_path, out_path)
    spec = dere.taskspec.load_from_xml(spec_path)
    model = MODELS[model_name](spec)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="bare_model.pkl")
@click.option("--outfile", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def train(corpus_path: str, model: str, outfile: str, corpus_format: str) -> None:
    _train(corpus_path, model, outfile, corpus_format)


def _train(
    corpus_path: str, model_path: str, out_path: str, corpus_format: str
) -> None:
    print("training with", corpus_path, model_path, out_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_io = CORPUS_IOS[corpus_format](model.spec)
    corpus = corpus_io.load(corpus_path)

    model.train(corpus)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
@click.option("--output-format", required=False, default=None)
@click.option("-o", required=True)
def predict(
    corpus_path: str,
    model: str,
    corpus_format: str,
    output_format: Optional[str],
    o: str,
) -> None:
    _predict(corpus_path, model, corpus_format, output_format, o)


def _predict(
    corpus_path: str,
    model_path: str,
    corpus_format: str,
    output_format: Optional[str],
    output_path: str,
) -> None:
    print("predicting with", corpus_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    input_corpus_io = CORPUS_IOS[corpus_format](model.spec)
    if output_format is None:
        output_format = corpus_format
    output_corpus_io = CORPUS_IOS[output_format](model.spec)

    corpus = input_corpus_io.load(corpus_path)

    model.predict(corpus)
    output_corpus_io.dump(corpus, output_path)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def evaluate(corpus_path: str, model: str, corpus_format: str) -> None:
    _evaluate(corpus_path, model, corpus_format)


def _evaluate(corpus_path: str, model_path: str, corpus_format: str) -> None:
    print("evaluating with", corpus_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_io = CORPUS_IOS[corpus_format](model.spec)
    corpus = corpus_io.load(corpus_path)

    predictions = model.predict(corpus)
    result = model.eval(corpus, predictions)
    print(result)  # or something smarter


if __name__ == "__main__":
    cli()
