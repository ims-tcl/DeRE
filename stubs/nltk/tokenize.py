from typing import List, Union, Set, Dict

class TreebankWordTokenizer:
    def tokenize(
        self, text: str,
        convert_parentheses: bool = False,
        return_str: bool = False
    ) -> Union[str, List[str]]:
        ...
