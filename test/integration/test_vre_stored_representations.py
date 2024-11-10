import sys
import shutil
from pathlib import Path
from natsort import natsorted
import numpy as np
import pandas as pd
from vre.representations.color import RGB, HSV
from vre import VRE
from vre.utils import FakeVideo

def test_vre_stored_representation():
    video_path = Path("data/rgb/npz")
    shutil.rmtree("data/hsv", ignore_errors=True)
    shutil.rmtree("data/buildings", ignore_errors=True)
    new_tasks = get_new_dronescapes_tasks()

    raw_data = [np.load(f)["arr_0"] for f in natsorted(video_path.glob("*.npz"), key=lambda p: p.name)]
    assert all(x.shape == raw_data[0].shape for x in raw_data), f"Images shape differ in '{video_path}'"
    video = FakeVideo(np.array(raw_data, dtype=np.uint8), frame_rate=1)

    representations = {"rgb": (rgb := RGB("rgb")), "buildings": new_tasks["buildings"], "hsv": HSV("hsv", [rgb])}
    representations["hsv"].binary_format = "npz"
    representations["hsv"].image_format = "png"
    representations["buildings"].binary_format = "npz"
    representations["buildings"].image_format = "png"
    vre = VRE(video, representations)
    print(vre)

    res = vre.run(output_dir=Path("data/"), start_frame=0, end_frame=None, output_dir_exists_mode="skip_computed")
    

if __name__ == "__main__":
    test_vre_stored_representation()
