from typing import List, Optional

from spacy.tokens import Doc

class Language:
    def __call__(self, text: str, disable: Optional[List[str]] = None) -> Doc:
        ...
