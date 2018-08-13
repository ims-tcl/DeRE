from pathlib import Path
from typing import Union, List, Optional

from spacy.language import Language

def load(
    name: Union[str, Path],
    disabled: Optional[List[str]] = None
) -> Language:
    ...
