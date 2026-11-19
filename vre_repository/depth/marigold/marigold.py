#!/usr/bin/env python3
"""
Marigold VRE representation
Standaline usage: `./marigold.py input/ output/` where `input/` is a directory of png/jpg/jpeg images
"""
import sys
from pathlib import Path
import numpy as np
import torch as tr
from overrides import overrides
from tqdm.auto import tqdm
from PIL import Image

from vre_video import VREVideo
from vre.utils import MemoryData
from vre.representations.mixins import LearnedRepresentationMixin
from vre import ReprOut
from vre_repository.depth import DepthRepresentation
from vre_repository.utils import fetch_weights, vre_load_weights

try:
    from .marigold_impl import AutoencoderKL, DDIMScheduler, LCMScheduler, MarigoldPipeline, UNet2DConditionModel
except ImportError:
    from vre_repository.depth.marigold.marigold_impl import (AutoencoderKL, DDIMScheduler, LCMScheduler,
                                                             MarigoldPipeline, UNet2DConditionModel)

class Marigold(DepthRepresentation, LearnedRepresentationMixin):
    """Marigold VRE implementation"""
    def __init__(self, variant: str, denoising_steps: int, ensemble_size: int, processing_resolution: int,
                 seed: int | None = None, **kwargs):
        DepthRepresentation.__init__(self, min_depth=0, max_depth=1, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        assert variant in ("marigold-v1-0", "marigold-lcm-v1-0", "testing"), variant
        self.variant = variant
        self.denoising_steps = denoising_steps
        self.ensemble_size = ensemble_size
        self.processing_resolution = processing_resolution
        self.model: MarigoldPipeline | None = None
        self.seed = seed

    @staticmethod
    @overrides
    def get_weights_paths(variant: str | None = None) -> list[str]:
        assert variant is not None, variant
        vae_path = Path(__file__).parent / "weights/vae.pt"
        match variant:
            case "marigold-lcm-v1-0":
                unet_path = [
                    Path(__file__).parent / "weights/marigold-lcm-v1-0_unet.pt/0000.pt",
                    Path(__file__).parent / "weights/marigold-lcm-v1-0_unet.pt/0001.pt",
                    Path(__file__).parent / "weights/marigold-lcm-v1-0_unet.pt/0002.pt",
                    Path(__file__).parent / "weights/marigold-lcm-v1-0_unet.pt/0003.pt",
                ]
                return [vae_path, unet_path]
            case _: raise NotImplementedError(variant)
        assert variant is None, variant

    @overrides
    def vre_setup(self, load_weights: bool=True):
        assert self.setup_called is False
        unet = UNet2DConditionModel(**self._get_unet_cfg())
        vae = AutoencoderKL(**self._get_vae_cfg())
        if load_weights:
            assert self.variant != "testing"
            paths = fetch_weights(type(self).get_weights_paths(self.variant))
            vae.load_state_dict(tr.load(paths[0], map_location="cpu"))
            unet.load_state_dict(vre_load_weights(paths[1]))

        scheduler = LCMScheduler() if self.variant == "marigold-lcm-v1-0" else DDIMScheduler()
        self.model = MarigoldPipeline(unet=unet, vae=vae, scheduler=scheduler,
                                      scale_invariant=True, shift_invariant=True)
        self.model = self.model.to(self.device)
        self.setup_called = True

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        output = MemoryData([self._make_one_frame(frame) for frame in video[ixs]])
        return ReprOut(frames=video[ixs], key=ixs, output=output)

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

    @tr.no_grad
    def _make_one_frame(self, frame: np.ndarray):
        assert self.model is not None
        tr_rgb = tr.from_numpy(frame).permute(2, 0, 1)[None].to(self.device)
        generator = None
        if self.seed is not None:
            generator = tr.Generator(self.device)
            generator.manual_seed(self.seed)
        y_model = self.model(tr_rgb, denoising_steps=self.denoising_steps, ensemble_size=self.ensemble_size,
                             processing_res=self.processing_resolution, generator=generator)
        return y_model[0][..., None]

    def _get_unet_cfg(self) -> dict:
        block_out_channels = {
            "testing": [8, 16, 32, 32],
            "marigold-lcm-v1-0": [320, 640, 1280, 1280],
            "marigold-v1-0": [320, 640, 1280, 1280],
        }[self.variant]
        norm_num_groups = {
            "testing": 8,
            "marigold-lcm-v1-0": 32,
            "marigold-v1-0": 32,
        }[self.variant]
        return {"in_channels": 8, "out_channels": 4, "block_out_channels": block_out_channels,
                "heads": [5, 10, 20, 20], "norm_num_groups": norm_num_groups, "cross_attention_dim": 1024}

    def _get_vae_cfg(self) -> dict:
        block_out_channels = {
            "testing": [8, 16, 32, 32],
            "marigold-lcm-v1-0": [128, 256, 512, 512],
            "marigold-v1-0": [128, 256, 512, 512],
        }[self.variant]
        norm_num_groups = {
            "testing": 8,
            "marigold-lcm-v1-0": 32,
            "marigold-v1-0": 32,
        }[self.variant]
        return {"block_out_channels": block_out_channels, "norm_num_groups": norm_num_groups}

def main():
    """main fn"""
    input_dir = Path(sys.argv[1])
    variant = "marigold-lcm-v1-0" if len(sys.argv) == 3 else sys.argv[3]
    device = "cuda" if tr.cuda.is_available() else "cpu"
    output_dir = Path(sys.argv[2]) / f"{variant}/{device}"
    image_paths = [x for x in input_dir.iterdir() if x.suffix in [".jpg", ".jpeg", ".png"]]
    assert len(image_paths) > 0, f"{input_dir} is empty of supported images"
    Path(output_dir / "png").mkdir(exist_ok=True, parents=True)

    marigold = Marigold(variant=variant, denoising_steps=4 if variant=="marigold-lcm-v1-0" else 10,
                        ensemble_size=1, processing_resolution=768, name="marigold", dependencies=[], seed=42)
    marigold.device = device
    marigold.vre_setup()

    expected = {
        "5131.png": {"cpu": (0.34113926, 0.25890285), "cuda": (0.37277249, 0.25874364)},
        "demo1.jpg": {"cpu": (0.47739768, 0.28376475), "cuda": (0.46834466, 0.32319403)},
        "demo2.jpg": {"cpu": (0.4603674, 0.2731725), "cuda": (0.51732534, 0.27054304)}
    }

    for rgb_path in (pbar := tqdm(image_paths, leave=True)):
        pbar.set_description(f"Estimating depth: {rgb_path.name}")
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = marigold.compute(rgb[None], [0])
        if marigold.variant == "marigold-lcm-v1-0" and rgb_path.name in expected.keys() \
                and marigold.seed is not None and marigold.seed == 42 and str(device) in expected[rgb_path.name]:
            try:
                assert np.allclose(A := expected[rgb_path.name][str(device)][0], (B := depth.output[0].mean())), (A, B)
                assert np.allclose(A := expected[rgb_path.name][str(device)][1], (B := depth.output[0].std())), (A, B)
                print("Mean and std match!")
            except AssertionError as e:
                print(e)

        depth_rsz = marigold.resize(depth, rgb.shape[0:2])
        depth_img = marigold.make_images(depth_rsz)[0]
        Image.fromarray(depth_img).save(output_dir / "png" / f"{rgb_path.stem}_pred.png")

if __name__ == "__main__":
    main()
