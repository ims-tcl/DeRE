from pathlib import Path
from typing import Union, Any, List, Tuple, Optional, IO

def dump(
    value: Any,
    filename: Union[str, Path, IO[bytes]],
    compress: Union[int, bool, Tuple[str, int]] = False,
    protocol: Optional[int] = None,
    cache_size: Optional[int] = None
) -> List[str]:
    ...

def load(
    filename: Union[str, Path, IO[bytes]],
    mmap_mode: Optional[str] = None
) -> Any:
    ...
