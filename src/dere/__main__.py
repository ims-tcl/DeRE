import click
import pickle

# path hackery to get imports working as intended
import sys
import os

path = os.path.dirname(sys.modules[__name__].__file__)
path = os.path.join(path, "..")
sys.path.insert(0, path)


from dere.readers import CorpusReader, BRATCorpusReader, XML123CorpusReader
from dere.models import BaselineModel, PGModel


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
