import importlib
import json
from typing import Optional, Dict, Any, Type
import logging
import os
import pickle
import sys
import warnings


logger = logging.getLogger("dere")  # noqa
handler = logging.StreamHandler()  # noqa
handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
)  # noqa
logger.addHandler(handler)  # noqa
logger.propagate = False  # noqa

import click

# path hackery to get imports working as intended
path = os.path.dirname(sys.modules[__name__].__file__)  # noqa
path = os.path.join(path, "..")  # noqa
sys.path.insert(0, path)  # noqa


# filter warnings from importing sklearn and numpy.
# sklearn specifically forces warnings to be displayed, which we don't like.
# https://github.com/scikit-learn/scikit-learn/issues/2531
def warn(*args, **kwargs):  # noqa
    pass  # noqa


old_warn = warnings.showwarning  # noqa
warnings.showwarning = warn  # noqa

import dere.taskspec
from dere.taskspec import TaskSpecification
from dere.corpus_io import CorpusIO, BRATCorpusIO, CQSACorpusIO
from dere.models import Model, BaselineModel, NOPModel
from dere.corpus import Corpus
import dere.evaluation
from dere.evaluation import Result

# restore ability to use warnings
warnings.showwarning = old_warn

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
@click.option("--verbose", "-v", is_flag=True, help="Show debug info")
@click.option(
    "--quiet", "-q", count=True, help="Do less logging. Can be provided multiple times."
)
def cli(verbose: bool, quiet: int) -> None:
    if verbose and quiet:
        raise click.BadParameter(
            "Options --verbose and --quiet are mutually exclusive."
        )
    if quiet > 2:
        quiet = 2
    # Calculation of verbosity level: verbose --> -1 (DEBUG), quiet --> 1 or 2
    #                                 neither verbose nor quiet --> 0
    val = -verbose + quiet
    # indices:                                                   -1
    #                  0            1              2             (3)
    verbosity = [logging.INFO, logging.WARN, logging.ERROR, logging.DEBUG][val]
    logging.basicConfig(stream=sys.stderr, level=verbosity)
    if not verbose:
        warnings.simplefilter("ignore")
    sys.path.append(os.getcwd())


@cli.command()
@click.option("--task-spec", required=True)
@click.option("--model-spec", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(task_spec: str, model_spec: str, outfile: str) -> None:
    _build(task_spec, model_spec, outfile)


def _build(task_spec_path: str, model_spec_path: str, out_path: str) -> None:
    logger.info(
        "[main] Building model with task spec %s and model spec %s, outputting to %s",
        task_spec_path,
        model_spec_path,
        out_path,
    )
    task_spec = dere.taskspec.load_from_xml(task_spec_path)
    with open(model_spec_path) as sf:
        model_spec = json.load(sf)
        # TODO Sean: Is there a better way to do this?
        # I want to allow relative paths in the model spec to be relative to the model spec file's location
        model_spec['__path__'] = model_spec_path

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
    logger.info(
        "[main] Training on corpus %s with model %s, outputting to %s",
        corpus_path,
        model_path,
        out_path,
    )
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
    logger.info("[main] Predicting on corpus %s and model %s", corpus_path, model_path)
    model = load_model(model_path)

    input_corpus_io = CORPUS_IOS[corpus_format](model.task_spec)
    if output_format is None:
        output_format = corpus_format
    output_corpus_io = CORPUS_IOS[output_format](model.task_spec)

    corpus = input_corpus_io.load(corpus_path, False)

    model.predict(corpus)

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_corpus_io.dump(corpus, output_path, False)


@cli.command()
@click.option("--hypo", required=True)
@click.option("--gold", required=True)
@click.option("--task-spec", required=True)
@click.option("--corpus-format", required=True)
def evaluate(hypo: str, gold: str, task_spec: str, corpus_format: str) -> None:
    _evaluate(hypo, gold, task_spec, corpus_format)


def _evaluate(hypo_path: str, gold_path: str, task_spec_path: str, corpus_format: str) -> None:
    logger.info("evaluating %s against %s using task specification %s", hypo_path, gold_path, task_spec_path)
    task_spec = dere.taskspec.load_from_xml(task_spec_path)
    corpus_io = CORPUS_IOS[corpus_format](task_spec)
    hypo = corpus_io.load(hypo_path, True)
    gold = corpus_io.load(gold_path, True)
    result = dere.evaluation.evaluate(hypo, gold, task_spec)
    logger.info("\n" + result.report())  # newline to keep the pretty-printed table


cli()
