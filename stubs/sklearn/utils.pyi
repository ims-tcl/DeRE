from typing import Optional, Union, Any, Iterable, Sequence
  
from sklearn import _ArrayLike

import numpy as np

def shuffle(
	*arrays: _ArrayLike,
	random_state: Union[None, np.random.RandomState, int] = None,
	n_samples: Union[None, int] = None
) -> Sequence[_ArrayLike]:
	...
