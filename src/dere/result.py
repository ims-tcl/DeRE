class Result:
    def __init__(self, **metrics):
        for metric in metrics:
            setattr(self, metric, metrics[metric])

    def __str__(self) -> str:
        ...  # "toString"

    def __repr__(self) -> str:
        ...  # "print"

    def __sub__(self, other):  # Result) -> Result:
        ...  # "compare"
        # r1 = Result(...)
        # r2 = Result(...)
        # difference = r2 - r1



