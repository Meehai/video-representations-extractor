"""lovely_numpy/tensors reimplementation"""
import torch as tr
import numpy as np

def monkey_patch(cls=tr.Tensor):
    "Monkey-patch lovely features into `cls`"
    setattr(cls, "v", cls.__repr__)
    cls.__repr__ = cls.__str__ = lo

def lo(x: np.ndarray | tr.Tensor | None) -> str:
    """reimplementation of lovely_numpy's lo() without any extra stuff that spams the terminal. Only for numericals!"""
    def _dt(x: tr.dtype | np.dtype) -> str:
        y = str(x).removeprefix("torch.") if str(x).startswith("torch.") else str(x)
        return y if y == "bool" else (f"{y[0]}{y[-1]}" if y[-1] == "8" else f"{y[0]}{y[-2:]}")
    assert isinstance(x, (type(None), np.ndarray, tr.Tensor)), type(x)
    if x is None:
        return x
    _r = lambda x: round(x.item(), 2) # pylint: disable=unnecessary-lambda-assignment
    arr_type = "arr" if isinstance(x, np.ndarray) else "tr"
    μ, σ = x.float().mean() if arr_type == "tr" else x.mean(), x.float().std() if arr_type == "tr" else x.std() # thx tr
    return f"{arr_type}{[*x.shape]} {_dt(x.dtype)} x∈[{_r(x.min())}, {_r(x.max())}], μ={_r(μ)}, σ={_r(σ)}"
