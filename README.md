# DeRE ![build status](https://travis-ci.com/ims-tcl/DeRE.svg?branch=master)


## Setup

### Requirements

- `Python 3.7+`
- `git`

### Installing DeRE

To install (as user):

    $ pip install .

To install (as developer):

    $ pip install -e .  # editable
    $ pip install -r dev_requirements.txt

To use DeRE, refer to the help that can be shown by specifying a `--help` flag either after the main command, or a subcommand (e.g. `dere build --help`):

    $ dere --help
    Usage: dere [OPTIONS] COMMAND [ARGS]...

    Options:
      -v, --verbose  Show debug info
      -q, --quiet    Do less logging. Can be provided multiple times.
      --help         Show this message and exit.

    Commands:
      build
      evaluate
      predict
      train

See also the [tutorials](#tutorials).


## Paper
[DeRE: A Task and Domain-Independent Slot Filling Framework for Declarative Relation Extraction](http://aclweb.org/...)


### Reference
If you plan to use DeRE please cite:


    @inproceedings{Adel2018,
      author = {Heike Adel and Laura Ana Maria Bostan and Sean Papay and Sebastian Padó and Roman Klinger},
      title = {{DeRE}: A Task and Domain-Independent Slot Filling Framework for Declarative Relation Extraction},
      booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
      optpages = {},
      year = {2018},
      address = {Brussels, Belgium},
      month = {October, November},
      publisher = {Association for Computational Linguistics},
      note = {accepted},
      pdf = {http://www.romanklinger.de/publications/dere2018.pdf}
    }

## Tutorials:

### User

In this tutorial we show how you can use a pretrained model for an existing task (i.e. [BioNLP'09 Shared Task on Event Extraction](http://www.nactem.ac.uk/tsujii/GENIA/SharedTask/index.shtml)) to obtain predictions on an unlabeled dataset.

You have:

* the BioNLP task already modeled at `task-specs/bionlpst.xml`
* a pretrained model called `baseline_trained.pkl` located at `tutorial/model/baseline_trained.pkl`
* an unlabeled corpus (in the BRAT format) located at `tutorial/data/test`

To use the pretrained model to generate predictions on the unlabeled corpus, and output them in the BRAT format at `tutorial/data/predict`, type the following command in your terminal:

    $ python3 dere predict --model-path tutorial/model/baseline_trained.pkl --corpus-format BRAT --corpus-path tutorial/data/test --output tutorial/data/predict/

You can check the general usage for `predict` by running:

    $ python3 dere predict --help


### Application Developer

In this tutorial we show you how to formalize an abstract conceptualization of an Information Extraction task  (i.e. [BioNLP'09 Shared Task on Event Extraction](http://www.nactem.ac.uk/tsujii/GENIA/SharedTask/index.shtml)), construct a model to model this task, train said model on a training set, and evaluate it on a test set of the corpus.

You  have:

* a labeled corpus split in train/test sets, located at `tutorial/data/(train|test)`
* an XML task sepcification located at `task-specs/bionlpst.xml`


Then you use

    $ dere build
    $ mkdir tutorial/model
    $ python3 dere build --task-spec task-specs/bionlpst.xml --model-spec model-specs/bionlpst-baseline.json --outfile tutorial/model/baseline.pkl

This will create a new, untrained model, which will be stored in the file `tutorial/model/baseline.pkl`.

To train the model on the training corpus you run:

    $ python3 dere train --model-path tutorial/model/baseline.pkl --corpus-format BRAT --outfile tutorial/model/baseline_trained.pkl --corpus-path tutorial/data/train

The trained model `baseline_trained.pkl` can be now evaluated on the test corpus by first predicting
the frames using the `predict` command as in:

    $ python3 dere predict --model-path tutorial/model/baseline_trained.pkl --corpus-format BRAT --corpus-path tutorial/data/test --output tutorial/data/predict/

The predicted annotations for the unlabeled set you find in the text files that end with `.ann` located at `tutorial/data/predict/`.

In order to evaluate the predictions you could use the `evaluate` command by running:

    $ python3 dere evaluate --predicted tutorial/data/predict --gold tutorial/data/test --task-spec task-specs/bionlpst.xml --corpus-format BRAT


You can check the general usage for `evaluate` by running:

    $ python3 dere evaluate --help

If you want to model your own task, you first need to specify your new task by writing it as an XML task sepcification. You can do that by following some examples of existing task specification files. These can be found in `task-specs/` in the DeRe repository. Then you will have to save this file as `task-specs/your_awesome_spec.xml`.

The other `dere` commands for work as exemplified already above on the BioNLP task!


### Model Developer

In order to implement a novel model and use it with-in `dere` do the following:

- write a class that subclasses `dere.models.Model`, e.g.:

```python
#!/usr/bin/env python

from dere.models import Model

class TutorialModel(Model):

    def train(self, corpus, dev_corpus=None):
        pass

    def predict(self, corpus):
        pass
```

Save this file as a python script, for example as `tutorial_model.py` and
let it be located at `dere/models`.

- the new Model has to have implemented at least two methods: `train`, `predict`, so implement them
- `train` gets a `Corpus` as the first argument and optionally another `Corpus`
  as second argument (a development corpus)
- `predict` gets a single `Corpus`
- both `train` and `predict` do not return anything: `predict` modifies the
  given corpus to add annotations, while `train` trains the model's classifier.

To work with your new model within `dere` you can use the already-introduced interface and specify your model class during the `build` step as a "dotted name" e.g. `tutorial_model.TutorialModel` (so filename of the module, without the `.py` extension + "." + name of the implemented class).

Again, to `build` the new model use:

    $ python3 dere build tutorial_model.TutorialModel --task-spec task-specs/bionlpst.xml --outfile tutorial/model/tutorial.pkl

The rest of the commands work as introduced for [User](#user) and [Application Developer](#application-developer).
