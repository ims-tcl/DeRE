from typing import List, Tuple, Union, Set, Dict, Iterable

class TreebankWordTokenizer:
    def tokenize(
        self, text: str,
        convert_parentheses: bool = False,
        return_str: bool = False
    ) -> Union[str, List[str]]:
        ...

    def span_tokenize(
        self,
        text: str
    ) -> Iterable[Tuple[int, int]]:
        ...
