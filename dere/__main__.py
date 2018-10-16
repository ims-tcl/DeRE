import importlib
import logging
import os
import pickle
import sys
import warnings
from typing import Optional

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
from dere.corpus_io import CorpusIO, BRATCorpusIO, CQSACorpusIO
from dere.models import BaselineModel, NOPModel
from dere.corpus import Corpus

# restore ability to use warnings
warnings.showwarning = old_warn


CORPUS_IOS = {"BRAT": BRATCorpusIO, "CQSA": CQSACorpusIO}

MODELS = {"baseline": BaselineModel, "nop": NOPModel}


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


@cli.command()
@click.option("--model", default="baseline")
@click.option("--spec", required=True)
@click.option("--outfile", default="bare_model.pkl")
def build(model: str, spec: str, outfile: str) -> None:
    _build(model, spec, outfile)


def _build(model_name: str, spec_path: str, out_path: str) -> None:
    logger.info(
        "[main] Building with model %s, specification %s, outputting to %s",
        model_name,
        spec_path,
        out_path,
    )
    spec = dere.taskspec.load_from_xml(spec_path)
    try:
        model = MODELS[model_name](spec)
    except KeyError:
        module, _, class_ = model_name.rpartition(".")
        model_module = importlib.import_module(module)
        model = getattr(model_module, class_)(spec)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


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
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_io = CORPUS_IOS[corpus_format](model.spec)
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
    logger.info("[main] Predicting on corpus %s and model %s", corpus_path, model_path)
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
    logger.info("[main] Evaluating on corpus %s and model %s", corpus_path, model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    corpus_io = CORPUS_IOS[corpus_format](model.spec)
    predictions = corpus_io.load(corpus_path, False)
    gold = corpus_io.load(corpus_path, True)

    model.predict(predictions)
    result = model.eval(gold, predictions)
    logger.info("[main] Result: %r", result)  # or something smarter


cli()
