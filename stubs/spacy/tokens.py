from typing import Iterator, Any, Union, List

from spacy.vocab import Vocab

class Token:
    text: str
    text_with_ws: str
    whitespace_: str
    orth: int
    orth_: str
    vocab: Vocab
    doc: Doc
    head: Token
    left_edge: Token
    right_edge: Token
    i: int
    ent_type: int
    ent_type_: str
    ent_iob: int
    ent_iob_: str
    ent_id: int
    ent_id_: str
    lemma: int
    lemma_: str
    norm: int
    norm_: str
    lower: int
    lower_: str
    shape: int
    shape_: str
    prefix: int
    prefix_: str
    suffix: int
    suffix_: str
    is_alpha: bool
    is_ascii: bool
    is_digit: bool
    is_lower: bool
    is_upper: bool
    is_title: bool
    is_punct: bool
    is_left_punct: bool
    is_right_punct: bool
    is_space: bool
    is_bracket: bool
    is_quote: bool
    is_currency: bool
    like_url: bool
    like_num: bool
    like_email: bool
    is_oov: bool
    is_stop: bool
    pos: int
    pos_: str
    tag: int
    tag_: str
    dep: int
    dep_: str
    lang: int
    lang_: str
    prob: float
    idx: int
    sentiment: float
    lex_id: int
    rank: int
    cluster: int
    _: Any
    
    


class Doc:
    def __iter__(self) -> Iterator[Token]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[Token, List[Token]]:
        ...
