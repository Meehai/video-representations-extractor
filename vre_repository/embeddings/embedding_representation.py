"""embedding_representation.py - embeddings: one (D, ) float vector per frame, or a (h, w, D) grid of them"""
from abc import ABC
import numpy as np
from overrides import overrides
from vre.utils import ReprOut
from vre.representations import Representation
from vre.representations.mixins import NpIORepresentationMixin
from vre_repository.utils import image_resize_batch

class EmbeddingRepresentation(NpIORepresentationMixin, Representation, ABC):
    """
    An embedding: one (D, ) float vector per frame, or a (h, w, D) grid of them (e.g. patch tokens). resize() keeps
    the embeddings as they are (only output_images are resized). make_images() draws a (D, ) vector as a grayscale grid
    at the frame's size; grid embeddings override it. Subclasses implement n_channels (== D).
    """
    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        emb = np.asarray(data.output)
        assert emb.ndim == 2 and emb.shape[1] == self.n_channels, (emb.shape, self.n_channels)
        grid_h = _grid_height(emb.shape[1])
        grid = emb.reshape(len(emb), grid_h, emb.shape[1] // grid_h)
        lo, hi = grid.min(axis=(1, 2), keepdims=True), grid.max(axis=(1, 2), keepdims=True)
        images = ((grid - lo) / np.maximum(hi - lo, 1e-8) * 255).astype(np.uint8)
        images = np.repeat(images[..., None], 3, axis=-1)
        if data.frames is None: # vre_reader: no frames, so the grid stays at its own size
            return images
        return image_resize_batch(images, *data.frames[0].shape[0:2], interpolation="nearest")

    @overrides
    def resize(self, data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        output_images = None
        if data.output_images is not None:
            output_images = image_resize_batch(data.output_images, *new_size, interpolation="nearest")
        return ReprOut(frames=data.frames, output=data.output, key=data.key, extra=data.extra,
                       output_images=output_images)

def _grid_height(d: int) -> int:
    """the largest divisor of d that is <= sqrt(d), so the grid is as square as possible (768 -> 24x32)"""
    assert d > 0, d
    h = int(d ** 0.5)
    while d % h != 0:
        h -= 1
    return h
