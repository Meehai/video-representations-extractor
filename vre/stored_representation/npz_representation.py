"""NpzRepresentation -- stored representation as npz files"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from overrides import overrides
import numpy as np
import torch as tr

from .stored_represntation import StoredRepresentation
from ..logger import vre_logger as logger
from ..utils import FixedSizeOrderedDict

_CACHE = FixedSizeOrderedDict(maxlen=1024)

class NpzRepresentation(StoredRepresentation):
    """Generic Task with data read from/saved to npz files. Tries to read data as-is from disk and store it as well"""
    def __init__(self, name: str, n_channels: int, dependencies: list[NpzRepresentation] | None = None):
        super().__init__(name, n_channels, dependencies)

    @overrides
    def load_from_disk(self, path: Path) -> tr.Tensor:
        """Reads the npz data from the disk and transforms it properly"""
        if (key := (self.name, str(getattr(self, "normalization", None)), path.name)) in _CACHE:
            logger.debug2(f"HIT: '{key}'")
            return deepcopy(_CACHE[key])
        logger.debug2(f"MISS: '{key}'")
        data = np.load(path, allow_pickle=False)
        data = data if isinstance(data, np.ndarray) else data["arr_0"] # in case on npz, we need this as well
        data = data.astype(np.float32) if np.issubdtype(data.dtype, np.floating) else data # float16 is dangerous
        res = tr.from_numpy(data)
        res = res.unsqueeze(-1) if len(res.shape) == 2 and self.n_channels == 1 else res # (H, W) in some dph/edges
        assert ((res.shape[-1] == self.n_channels and len(res.shape) == 3) or
                (len(res.shape) == 2 and self.is_classification)), f"{self.name}: {res.shape} vs {self.n_channels}"
        _CACHE[key] = deepcopy(res)
        return self.return_fn(res)

    @overrides
    def save_to_disk(self, data: tr.Tensor, path: Path):
        """stores this item to the disk which can then be loaded via `load_from_disk`"""
        np.save(path, data.cpu().detach().numpy(), allow_pickle=False)
