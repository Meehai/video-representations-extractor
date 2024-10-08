#!/usr/bin/env python3
"""
Marigold VRE representation
Standaline usage: `./marigold.py input/ output/` where `input/` is a directory of png/jpg/jpeg images
"""
import sys
from pathlib import Path
import numpy as np
import torch
from overrides import overrides
from tqdm.auto import tqdm
from vre.utils import RepresentationOutput, image_resize_batch, image_read, colorize_depth_maps, image_write
from vre.representations.depth.marigold.marigold_impl import MarigoldPipeline
from vre.representation import Representation

class Marigold(Representation):
    """Marigold VRE implementation"""
    def __init__(self, variant: str, denoising_steps: int, ensemble_size: int, processing_resolution: int, **kwargs):
        super().__init__(**kwargs)
        assert variant in ("prs-eth/marigold-v1-0", "prs-eth/marigold-lcm-v1-0"), variant
        self.variant = variant
        self.denoising_steps = denoising_steps
        self.ensemble_size = ensemble_size
        self.processing_resolution = processing_resolution
        self.model: MarigoldPipeline | None = None

    @overrides
    def vre_setup(self):
        self.model = MarigoldPipeline.from_pretrained(self.variant)

    @torch.no_grad
    def _make_one_frame(self, frame: np.ndarray):
        tr_rgb = torch.from_numpy(frame).permute(2, 0, 1)[None]
        (gen := torch.Generator()).manual_seed(42)
        return self.model(tr_rgb, denoising_steps=self.denoising_steps, # pylint: disable=not-callable
                          ensemble_size=self.ensemble_size, processing_res=self.processing_resolution, generator=gen)

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        res = np.stack([self._make_one_frame(frame) for frame in frames])
        return RepresentationOutput(output=res)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        depth_colored = colorize_depth_maps(repr_data.output, 0, 1)
        return (depth_colored * 255).astype(np.uint8)

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return RepresentationOutput(output=image_resize_batch(repr_data.output, *new_size, "bilinear").clip(0, 1))

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[0:2]

def main():
    """main fn"""
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    image_paths = [x for x in input_dir.iterdir() if x.suffix in [".jpg", ".jpeg", ".png"]]
    assert len(image_paths) > 0, f"{input_dir} is empty of supported images"
    Path(output_dir / "png").mkdir(exist_ok=True, parents=True)

    marigold = Marigold("prs-eth/marigold-lcm-v1-0", denoising_steps=4, ensemble_size=1, processing_resolution=768,
                        name="marigold", dependencies=[])
    marigold.vre_setup()

    expected = {
        "5131.png": (0.34113926, 0.25890285),
        "demo1.jpg": (0.47739768, 0.28376475),
        "demo2.jpg": (0.4603674, 0.2731725),
    }

    for rgb_path in (pbar := tqdm(image_paths, leave=True)):
        pbar.set_description(f"Estimating depth: {rgb_path.name}")
        rgb = image_read(rgb_path, "PIL")
        depth = marigold(rgb[None])
        if rgb_path.name in expected.keys():
            assert np.allclose(A := expected[rgb_path.name][0], (B := depth.output[0].mean())), (A, B)
            assert np.allclose(A := expected[rgb_path.name][1], (B := depth.output[0].std())), (A, B)
            print("Mean and std match!")

        depth_rsz = marigold.resize(depth, rgb.shape[0:2])
        depth_img = marigold.make_images(None, depth_rsz)[0]
        image_write(depth_img, output_dir / "png" / f"{rgb_path.stem}_pred.png")

if __name__ == "__main__":
    main()
