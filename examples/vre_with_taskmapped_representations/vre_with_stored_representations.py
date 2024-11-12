import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / "semantic_mapper"))
from semantic_mapper import get_new_dronescapes_tasks, run_vre

import pandas as pd
from vre import VRE
from vre.representations.color import RGB, HSV
from vre.utils import FFmpegVideo

def main():
    video_path = Path(sys.argv[1])
    vre_path = Path(__file__).parent / video_path.name
    assert not vre_path.exists(), f"{vre_path} exists, delete first!"
    frames = [0, 100, 1000]
    run_vre(video_path, vre_path, frames=frames) # TODO: get rid of this and make VRE deepwalk if needed.

    new_tasks = get_new_dronescapes_tasks()
    representations = {"rgb": (rgb := RGB("rgb")), "buildings": new_tasks["buildings"], "hsv": HSV("hsv", [rgb])}
    video = FFmpegVideo(video_path)
    vre = VRE(video, representations) \
        .set_io_parameters(binary_format="npz", image_format="png", compress=True) \
        .set_compute_params(output_size="video_shape", batch_size=3)
    print(vre)
    res = vre.run(output_dir=vre_path, frames=frames, output_dir_exists_mode="overwrite")
    print(pd.DataFrame(res["run_stats"], index=res["runtime_args"]["frames"]))
    _ = vre.run(output_dir=vre_path, frames=frames, output_dir_exists_mode="skip_computed")

if __name__ == "__main__":
    main()
