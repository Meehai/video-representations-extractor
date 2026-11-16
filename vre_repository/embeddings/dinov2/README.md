# DINOv2 in VRE

Per-frame embeddings from [DINOv2](https://github.com/facebookresearch/dinov2), as `embeddings/dinov2`: one global
L2-normalized vector, or the raw patch tokens (`pool: none`). Pure torch: `dinov2_impl.py` is a port of transformers' `Dinov2Model` (no `transformers` dep).

```yaml
dinov2:
  type: embeddings/dinov2
  dependencies: []
  parameters:
    variant: dinov2-base  # dinov2-small (D=384) | dinov2-base (D=768) | dinov2-large (D=1024)
    pool: cat             # cls (CLS token) | gem (GeM p=3 over patch tokens) | cat (both, normalized; 2*D)
                          # | none (no pooling: the patch tokens, not normalized)
    input_size: [378, 672] # (h, w) every frame is resized to (aspect not kept), both multiples of 14
```

`input_size` is part of what the output means: the same frame at another size gives other vectors. The default
`[378, 672]` is a 16:9 frame with its long side at 672; set it to your video's aspect to avoid squashing.

Output: `npz/<frame>.npz`, key `arr_0`, float32:
- `cls | gem | cat`: `(D,)` or `(2*D,)` for `cat`, unit norm. Images: each vector drawn as a grayscale grid.
- `none`: `(h/14, w/14, D)`, the last layer's patch tokens (after the final layernorm, row-major, no CLS), not
  normalized. This is what dense matchers (e.g. RoMa's coarse stage) consume; the caller does the rest (its own
  resizing, fine CNN features, ...). Images: the top-3 principal components of the frame's tokens as RGB (the
  basis is per frame, so colors are not comparable across frames).
  Size: `h/14 * w/14 * D * 4` bytes per frame, e.g. 40x40x1024 (dinov2-large at 560x560) = 6.5MB.

`output_size` never touches the embeddings, only the images (resized to the frame's size).
On cuda the model runs under float16 autocast.

## Weights

`weights/<variant>.pt` is transformers' `facebook/<variant>` state_dict as-is (same keys), so the two stay
interchangeable: `Dinov2Model.load_state_dict(...)` works both ways. Fetched on first use, like every other VRE
representation: from VRE's gitlab, and if it isn't there (HTTP/network error), from huggingface's
`facebook/<variant>/pytorch_model.bin` at the commit pinned in `HF_REVISIONS` (`dinov2.py`). That file is the very same
state_dict, tensor for tensor. To regenerate them (needs `transformers` once, not at runtime):

```python
import torch
from transformers import AutoModel
for variant in ["dinov2-small", "dinov2-base", "dinov2-large"]:
    torch.save(AutoModel.from_pretrained(f"facebook/{variant}").state_dict(), f"weights/{variant}.pt")
```

`weights/` is git-ignored: add new ones with `git add -f` (they are git-lfs, see `.gitattributes`).

## Compatibility with transformers

Same outputs as `transformers.AutoModel.from_pretrained("facebook/dinov2-*")` (>= 4.45, default sdpa attention),
bit-identical on CPU (fp32) and cuda (fp16 autocast), for any input size. Two details matter:
- position embeddings: bicubic-interpolated to the patch grid by `size=` (transformers >= 4.45). Older versions
  (<= 4.44) used `scale_factor` with a +0.1 offset, which gives ~4% different embeddings on non-518x518 inputs.
- attention: `F.scaled_dot_product_attention` (transformers' default `sdpa`). The `eager` attention differs by ~1e-6.

Check it with `test/manual/dinov2/check_hf_parity.py` (needs `transformers`); it also compares against an Embedder
that wraps transformers with the same preprocessing and pooling.
