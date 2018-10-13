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
import dere.evaluation
from dere.evaluation import Result

CORPUS_IOS = {"BRAT": BRATCorpusIO, "CQSA": CQSACorpusIO}


def instantiate_model(task_spec: TaskSpecification, model_spec: Dict[str, Any]) -> Model:
    model_type = model_spec['model_type']
    if "." not in model_type:
        model_type = "dere.models.%s" % model_type
    module, _, class_ = model_type.rpartition(".")
    model_module = importlib.import_module(module)
    model_class = getattr(model_module, class_)
    params = model_spec.get('params', {})
    model = model_class(task_spec, model_spec, **params)
    assert isinstance(model, Model)
    return model


def load_model(path: str) -> Model:
    with open(path, 'rb') as f:
        task_spec, model_spec = pickle.load(f)
        model = instantiate_model(task_spec, model_spec)
        model.load(f)
        return model


def save_model(model: Model, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump((model.task_spec, model.model_spec), f)
        model.dump(f)


@click.group()
@click.option("--verbosity", default="INFO")
def cli(verbosity: str) -> None:
    sys.path.append(os.getcwd())
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, verbosity))


@cli.command()
@click.option("--task-spec", required=True)
@click.option("--model-spec", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(task_spec: str, model_spec: str, outfile: str) -> None:
    _build(task_spec, model_spec, outfile)


def _build(task_spec_path: str, model_spec_path: str, out_path: str) -> None:
    print("building with", task_spec_path, model_spec_path, out_path)
    task_spec = dere.taskspec.load_from_xml(task_spec_path)
    with open(model_spec_path) as sf:
        model_spec = json.load(sf)

    model = instantiate_model(task_spec, model_spec)
    model.initialize()
    save_model(model, out_path)


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
    model = load_model(model_path)

    corpus_io = CORPUS_IOS[corpus_format](model.task_spec)
    corpus = corpus_io.load(corpus_path, load_gold=True)

    dev_corpus: Optional[Corpus] = None
    if dev_corpus_path is not None:
        dev_corpus = corpus_io.load(dev_corpus_path, load_gold=True)
    elif corpus_split is not None:
        ratio = float(corpus_split)
        corpus, dev_corpus = corpus.split(ratio)

    model.train(corpus, dev_corpus)
    save_model(model, out_path)


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
    model = load_model(model_path)

    input_corpus_io = CORPUS_IOS[corpus_format](model.task_spec)
    if output_format is None:
        output_format = corpus_format
    output_corpus_io = CORPUS_IOS[output_format](model.task_spec)

    corpus = input_corpus_io.load(corpus_path, False)

    model.predict(corpus)

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_corpus_io.dump(corpus, output_path)


@cli.command()
@click.option("--hypo", required=True)
@click.option("--gold", required=True)
@click.option("--task-spec", required=True)
@click.option("--corpus-format", required=True)
def evaluate(hypo: str, gold: str, task_spec: str, corpus_format: str) -> None:
    _evaluate(hypo, gold, task_spec, corpus_format)


def _evaluate(hypo_path: str, gold_path: str, task_spec_path: str, corpus_format: str) -> None:
    print("evaluating %s against %s using task specification %s" % (hypo_path, gold_path, task_spec_path))

    task_spec = dere.taskspec.load_from_xml(task_spec_path)
    corpus_io = CORPUS_IOS[corpus_format](task_spec)
    hypo = corpus_io.load(hypo_path, True)
    gold = corpus_io.load(gold_path, True)

    result = dere.evaluation.evaluate(hypo, gold, task_spec)
    print(result.report())


cli()
