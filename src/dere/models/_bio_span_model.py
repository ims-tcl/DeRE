from typing import List, Any, Optional, Tuple, Dict

from dere.taskspec import TaskSpecification, SpanType
from dere.corpus import Corpus, Instance
from dere.models import Model


class BIOSpanModel(Model):

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus] = None) -> None:
        x, ys = self._corpus_xys(corpus)
        if dev_corpus is not None:
            dev_x, dev_ys = self._corpus_xys(dev_corpus)
            for span_type in self.spec.span_types:
                self.sequence_train(span_type, x, ys[span_type], dev_x, dev_ys[span_type])
        else:
            for span_type in self.spec.span_types:
                self.sequence_train(span_type, x, ys[span_type])

    def predict(self, corpus: Corpus) -> None:
        x, _ = self._corpus_xys(corpus)
        for span_type in self.spec.span_types:
            predictions = self.sequence_predict(span_type, x)
            for instance, prediction in zip(corpus.instances, predictions):
                token_spans = self.span_tokenize(instance.text)
                self._make_spans_from_labels(instance, span_type, prediction)

    """
    Train the sequence labeler for the given span type. If development data is provided, it may be used to
    tune hyperparameters. Subclasses should implement this method.

    Args:
        span_type: The span type whose classifier is to be trained.
        x: Classifier input -- a list of token sequences.
        y: Gold-standard labels -- a list of BIO sequences.
        x_dev: If provided, classifier input for the development set.
        y_dev: If provided, BIO labels for the development set.
    """
    def sequence_train(
        self,
        span_type: SpanType,
        x: List[List[str]],
        y: List[List[str]],
        x_dev: Optional[List[List[str]]] = None,
        y_dev: Optional[List[List[str]]] = None,
    ) -> None:
        ...

    """
    Generate BIO predictions for the given span type, provided input data. This should generally only be
    called after train() has been callled for the corresponding span type. Subclasses should implement this
    method.

    Args:
        span_type: The span type for which predictions should be generated.
        x: Classifier input -- a list of token sequences.

    Returns:
        A list of BIO label sequences.
    """
    def sequence_predict(self, span_type: SpanType, x: List[List[str]]) -> List[List[str]]:
        ...

    """
    Tokenize a piece of text. Subclasses should implement this method.

    Args:
        text: The text to tokenize.

    Returns:
        A list of (left, right) index pairs describing token boundaries.  Each token consists of the text
        between left (inclusive) and right (exclusive). Tokens should be non-overlapping (but possibly with
        gaps between tokens) and monotonic.
    """
    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        ...

    """
    Normalize a token before providing it to the sequence labeler. The default implementation returns the
    token unchanged. Subclasses may override this method to e.g. lowercase all tokens before they are sent
    to the sequence labeler.

    Args:
        token: The token text to be normalized.

    Returns:
        The normalized token.
    """
    def normalize_token(self, token: str) -> str:
        return token

    def _instance_xys(self, instance: Instance) -> Tuple[List[str], Dict[SpanType, List[str]]]:
        token_spans = self.span_tokenize(instance.text)
        tokens = [self.normalize_token(instance.text[l:r]) for (l, r) in token_spans]
        labels = {span_type: ["O"] * len(tokens) for span_type in self.spec.span_types}
        for span in instance.spans:
            type_specific_labels = labels[span.span_type]
            span_begun = False
            for i, (left, right) in enumerate(token_spans):
                # if the token span ends before the annotated span begins
                if right <= span.left:
                    continue
                # if the token span begins after the annotated span ends:
                elif left >= span.right:
                    break
                elif not span_begun:
                    type_specific_labels[i] = "B"
                    span_begun = True
                else:
                    type_specific_labels[i] = "I"
        return tokens, labels

    def _corpus_xys(self, corpus: Corpus) -> Tuple[List[List[str]], Dict[SpanType, List[List[str]]]]:
        x: List[List[str]] = []
        ys: Dict[SpanType, List[List[str]]] = {}
        for instance in corpus.instances:
            instance_x, instance_ys = self._instance_xys(instance)
            x.append(instance_x)
            for span_type in self.spec.span_types:
                ys[span_type].append(instance_ys[span_type])
        return x, ys

    def _make_spans_from_labels(
        self,
        instance: Instance,
        span_type: SpanType,
        labels: List[str],
        strict: bool = True
    ) -> None:
        token_spans = self.span_tokenize(instance.text)
        current_span_left = None
        last_token_right = None
        for bio, (l, r) in zip(labels, token_spans):
            if current_span_left is not None and bio != 'I':
                instance.new_span(span_type, current_span_left, last_token_right)
                current_span_left = None
                if bio == 'B' or (not strict and bio == 'I' and current_span_left is None):
                    current_span_left = l
                last_token_right = r
