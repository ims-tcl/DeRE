from typing import Optional, List

class CRF:
    def __init__(
        self,
        algorithm: str = 'lbfgs',
        min_freq: float = 0,
        all_possible_states: bool = False,
        all_possible_transitions: bool = False,
        c1: float = 0,
        c2: float = 1,
        max_iterations: Optional[int] = None,
        num_memories: int = 6,
        epsilon: float = 1e-5,
        period: int =10,
        delta: float =1e-5,
        linesearch: str = 'MoreThuente',
        max_linesearch: int = 20,
        calibration_eta: float = 0.1,
        calibration_rate: float = 2.0,
        calibration_samples: int = 1000,
        calibration_candidates: int = 10,
        calibration_max_trials: int = 20,
        pa_type: int = 1,
        c: float = 1,
        error_sensitive: bool = True,
        averaging: bool = True,
        variance: float = 1,
        gamma: float = 1,
        verbose: bool = False,
        model_filename: Optional[str] = None
    ) -> None:
        ...

    def fit(
        self,
        X: List[List[dict]],
        y: List[List[str]],
        X_dev: Optional[List[List[dict]]] = None,
        y_dev: Optional[List[List[str]]] = None,
    ) -> CRF:
        ...

    def predict(self, X: List[List[dict]]) -> List[List[str]]:
        ...

