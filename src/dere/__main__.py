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
    print("building with", model, schema, outfile)
    mdl = MODELS[model](schema)
    with open(outfile, "wb") as f:
        pickle.dump(mdl, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="bare_model.pkl")
@click.option("--outfile", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def train(
    corpus_path: str, model: str, outfile: str, corpus_format: str
) -> None:
    print("training with", corpus_path, model, outfile)
    with open(model, "rb") as f:
        mdl = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    mdl.train(corpus)
    with open(outfile, "wb") as f:
        pickle.dump(mdl, f)


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def predict(corpus_path: str, model: str, corpus_format: str) -> None:
    print("predicting with", corpus_path, model)
    with open(model, "rb") as f:
        mdl = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    predictions = mdl.predict(corpus)
    print(predictions)  # or something smarter


@cli.command()
@click.argument("corpus_path")
@click.option("--model", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def evaluate(corpus_path: str, model: str, corpus_format: str) -> None:
    print("evaluating with", corpus_path, model)
    with open(model, "rb") as f:
        mdl = pickle.load(f)

    corpus_reader = CORPUS_READERS[corpus_format](corpus_path)
    corpus = corpus_reader.load()

    predictions = mdl.predict(corpus)
    result = mdl.eval(corpus, predictions)
    print(result)  # or something smarter


if __name__ == "__main__":
    cli()