"""lovely_numpy/tensors reimplementation"""
import numpy as np

try:
    import torch as tr
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

def monkey_patch(cls=None):
    "Monkey-patch lovely features into `cls`"
    if cls is None:
        if not _HAS_TORCH:
            return
        cls = tr.Tensor
    setattr(cls, "v", cls.__repr__)
    cls.__repr__ = cls.__str__ = lo

def lo(x: "np.ndarray | tr.Tensor | None") -> str:
    """reimplementation of lovely_numpy's lo() without any extra stuff that spams the terminal. Only for numericals!"""
    def _dt(x) -> str:
        y = str(x).removeprefix("torch.") if str(x).startswith("torch.") else str(x)
        return y if y == "bool" else (f"{y[0]}{y[-1]}" if y[-1] == "8" else f"{y[0]}{y[-2:]}")
    _valid_types = (type(None), np.ndarray, tr.Tensor) if _HAS_TORCH else (type(None), np.ndarray)
    assert isinstance(x, _valid_types), type(x)
    if x is None:
        return x
    _r = lambda x: round(x.item(), 2) # pylint: disable=unnecessary-lambda-assignment
    arr_type = "arr" if isinstance(x, np.ndarray) else "tr"
    μ, σ = x.float().mean() if arr_type == "tr" else x.mean(), x.float().std() if arr_type == "tr" else x.std() # thx tr
    return f"{arr_type}{[*x.shape]} {_dt(x.dtype)} x∈[{_r(x.min())}, {_r(x.max())}], μ={_r(μ)}, σ={_r(σ)}"
