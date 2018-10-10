import click
import pickle
import logging
import importlib
import json
from typing import Optional, Dict, Any, Type

# path hackery to get imports working as intended
import sys
import os

path = os.path.dirname(sys.modules[__name__].__file__)  # noqa
path = os.path.join(path, "..")  # noqa
sys.path.insert(0, path)  # noqa

from dere.corpus_io import CorpusIO, BRATCorpusIO, CQSACorpusIO
from dere.models import Model, BaselineModel, NOPModel
from dere.corpus import Corpus
import dere.taskspec
from dere.taskspec import TaskSpecification


CORPUS_IOS = {"BRAT": BRATCorpusIO, "CQSA": CQSACorpusIO}

MODELS: Dict[str, Type[Model]] = {"baseline": BaselineModel, "nop": NOPModel}


def instantiate_model(name: str, task_spec: TaskSpecification, model_spec: Model.ModelSpec) -> Model:
    try:
        return MODELS[name](task_spec, model_spec)
    except KeyError:
        module, _, class_ = name.rpartition(".")
        model_module = importlib.import_module(module)
        model_class = getattr(model_module, class_)
        model = model_class(task_spec, model_spec)
        assert isinstance(model, Model)
        return model


@click.group()
@click.option("--verbosity", default="INFO")
def cli(verbosity: str) -> None:
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, verbosity))


@cli.command()
@click.option("--model", default="baseline")
@click.option("--task-spec", required=True)
@click.option("--model-spec", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(model: str, task_spec: str, model_spec: str, outfile: str) -> None:
    _build(model, task_spec, model_spec, outfile)


def _build(model_name: str, task_spec_path: str, model_spec_path: str, out_path: str) -> None:
    print("building with", model_name, task_spec_path, model_spec_path, out_path)
    task_spec = dere.taskspec.load_from_xml(task_spec_path)
    with open(model_spec_path) as sf:
        model_spec = json.load(sf)
    model = instantiate_model(model_name, task_spec, model_spec)
    model.initialize()
    with open(out_path, "wb") as f:
        pickle.dump((model_name, task_spec, model_spec), f)
        # dump the model's (initialized) parameters
        model.dump(f)


@cli.command()
@click.option("--corpus-path", required=True)
@click.option("--model-path", default="bare_model.pkl")
@click.option("--outfile", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
@click.option("--dev-corpus", required=False)
@click.option("--corpus-split", required=False)
def train(
    corpus_path: str,
    model_path: str,
    outfile: str,
    corpus_format: str,
    dev_corpus: Optional[str],
    corpus_split: Optional[str],
) -> None:
    _train(corpus_path, model_path, outfile, corpus_format, dev_corpus, corpus_split)


def _train(
    corpus_path: str,
    model_path: str,
    out_path: str,
    corpus_format: str,
    dev_corpus_path: Optional[str],
    corpus_split: Optional[str],
) -> None:
    print("training with", corpus_path, model_path, out_path)
    with open(model_path, "rb") as f:
        (model_name, task_spec, model_spec) = pickle.load(f)
        model = instantiate_model(model_name, task_spec, model_spec)
        model.load(f)

    corpus_io = CORPUS_IOS[corpus_format](task_spec)
    corpus = corpus_io.load(corpus_path, load_gold=True)

    dev_corpus: Optional[Corpus] = None
    if dev_corpus_path is not None:
        dev_corpus = corpus_io.load(dev_corpus_path, load_gold=True)
    elif corpus_split is not None:
        ratio = float(corpus_split)
        corpus, dev_corpus = corpus.split(ratio)

    model.train(corpus, dev_corpus)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


@cli.command()
@click.option("--corpus-path", required=True)
@click.option("--model-path", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
@click.option("--output-format", required=False, default=None)
@click.option("--output", "-o", required=True)
def predict(
    corpus_path: str,
    model_path: str,
    corpus_format: str,
    output_format: Optional[str],
    output: str,
) -> None:
    _predict(corpus_path, model_path, corpus_format, output_format, output)


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

    corpus = input_corpus_io.load(corpus_path, False)

    model.predict(corpus)

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_corpus_io.dump(corpus, output_path)


@cli.command()
@click.argument("corpus_path")
@click.option("--model-path", default="trained_model.pkl")
@click.option("--corpus-format", required=True)
def evaluate(corpus_path: str, model_path: str, corpus_format: str) -> None:
    _evaluate(corpus_path, model_path, corpus_format)


def _evaluate(corpus_path: str, model_path: str, corpus_format: str) -> None:
    print("evaluating with", corpus_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_io = CORPUS_IOS[corpus_format](model.spec)
    corpus = corpus_io.load(corpus_path, False)
    gold = corpus_io.load(corpus_path, True)

    result = model.evaluate(corpus, gold)
    print(result)  # or something smarter


cli()
