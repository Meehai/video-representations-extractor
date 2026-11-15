# DINOv2 in VRE

One global, L2-normalized embedding per frame from [DINOv2](https://github.com/facebookresearch/dinov2), as
`embeddings/dinov2`. Pure torch: `dinov2_impl.py` is a port of transformers' `Dinov2Model` (no `transformers` dep).

```yaml
dinov2:
  type: embeddings/dinov2
  dependencies: []
  parameters:
    variant: dinov2-base  # dinov2-small (D=384) | dinov2-base (D=768) | dinov2-large (D=1024)
    pool: cat             # cls (CLS token) | gem (GeM p=3 over patch tokens) | cat (both, normalized; 2*D)
    long_side: 672        # resize, aspect kept, long side to this; both sides rounded to a multiple of 14
```

Output: `npz/<frame>.npz`, key `arr_0`, float32 `(D,)` or `(2*D,)` for `cat`, unit norm. `output_size` does not touch
the vectors (they have no spatial dims), only the images: each vector drawn as a grayscale grid at the frame's size.
On cuda the model runs under float16 autocast.

## Weights

`weights/<variant>.pt` is transformers' `facebook/<variant>` state_dict as-is (same keys), so the two stay
interchangeable: `Dinov2Model.load_state_dict(...)` works both ways. Fetched on first use, like every other VRE
representation. To regenerate them (needs `transformers` once, not at runtime):

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
