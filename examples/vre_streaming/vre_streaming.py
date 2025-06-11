#!/usr/bin/env python3
import torch as tr
import matplotlib.pyplot as plt
from pathlib import Path
from tempfile import TemporaryDirectory
from pprint import pprint
import os

from vre import VRE, FFmpegVideo, ReprOut, Representation
from vre.vre_runtime_args import VRERuntimeArgs
from vre.representations import IORepresentationMixin
from vre.representations.io_representation_mixin import ImageFormat
from vre.utils import get_project_root, collage_fn, image_write, random_chars

from vre_repository.color.rgb import RGB
from vre_repository.semantic_segmentation.safeuav import SafeUAV

os.environ["VRE_DEVICE"] = device = "cuda" if tr.cuda.is_available() else "cpu"

def display_imgs(res: dict[str, ReprOut]):
    if all(res[r].output_images is None for r in repr_names):
        print("Image format not set, not images were computed. Skipping.")
        return

    repr_names = list(res.keys())
    for i, frame_ix in enumerate(res[repr_names[0]].key):
        imgs = [res[r].output_images[i] if res[r].output_images is not None else None for r in repr_names]
        collage = collage_fn(imgs, titles=repr_names, size_px=70, rows_cols=None)
        image_write(collage, f"collage_{frame_ix}.png")
        plt.figure(figsize=(20, 10))
        plt.imshow(collage)
        plt.show()

def getitem1(vre: VRE, ix: list[int]) -> dict[str, ReprOut]:
    """wrapper since getitem cannot receive extra args. This is the old 'VREStreaming' code basically."""
    vre.set_io_parameters(binary_format="npz", compress=False, image_format="png", output_size="video_shape")
    output_dir = Path(TemporaryDirectory(prefix="vre_getitem").name)
    run_id = random_chars(n=10) # we need a temp run id for getitem in "streaming" mode

    res: dict[str, ReprOut] = {}
    for vre_repr in vre.representations:
        assert isinstance(vre_repr, IORepresentationMixin), (vre_repr, type(vre_repr))
        (output_dir / vre_repr.name).mkdir(exist_ok=True, parents=True)
        runtime_args = VRERuntimeArgs(video=vre.video, representations=vre.representations, frames=ix,
                                      exception_mode="stop_execution", n_threads_data_storer=1)
        _ = vre.do_one_representation(run_id=run_id, representation=vre_repr, output_dir=output_dir,
                                      output_dir_exists_mode="skip_computed", runtime_args=runtime_args)
        res[vre_repr.name] = vre._load_from_disk_if_possible(vre_repr, vre.video,
                                                              ixs=list(ix), output_dir=output_dir)
        if vre_repr.image_format != ImageFormat.NOT_SET:
            res[vre_repr.name].output_images = vre_repr.make_images(res[vre_repr.name])
    return res

def getitem2(vre: VRE, ix: list[int]) -> dict[str, ReprOut]:
    """trying to implement vre streaming w/o wrapping the batched functions"""
    return getitem1(vre, ix)

def main():
    video = FFmpegVideo(get_project_root() / "resources/test_video.mp4")
    print(video)

    representations: list[Representation] = [
        RGB(name="rgb", dependencies=[]),
        safeuav := SafeUAV(name="safeuav", dependencies=[], disk_data_argmax=False, variant="model_150k"),
    ]
    safeuav.device = device
    vre = VRE(video, representations)
    vre.to_graphviz().render("graph", format="png", cleanup=True)
    vre.set_compute_params(batch_size=1)
    frames = [100, 1000, 3000]

    res = getitem1(vre, frames)
    res2 = getitem2(vre, frames)
    for k in res.keys():
        assert res[k] == res2[k], (k, res[k], res2[k])
    # pprint(res)
    # display_imgs(res)

if __name__ == "__main__":
    main()
