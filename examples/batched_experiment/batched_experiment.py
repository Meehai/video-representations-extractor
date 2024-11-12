#!/usr/bin/env python3
"""experiment using batches of frames"""
import sys
from pathlib import Path
from omegaconf import OmegaConf
import shutil
import json
import pandas as pd

from vre import VRE
from vre.representations import build_representations_from_cfg
from vre.utils import get_project_root, FFmpegVideo

def main():
    """main fn"""
    video = FFmpegVideo(get_project_root() / "resources/test_video.mp4")
    representations = build_representations_from_cfg(OmegaConf.load(Path(__file__).parent / "cfg.yaml"))
    vre = VRE(video, representations)
    frames = list(range(1000, 1000 + (5 if len(sys.argv) == 1 else int(sys.argv[1]))))
    batch_sizes = [1, 3, 5]
    out_dirs = [Path(__file__).parent / f"experiment_{batch_size}" for batch_size in batch_sizes]

    for batch_size, out_dir in zip(batch_sizes, out_dirs):
        shutil.rmtree(out_dir, ignore_errors=True)
        vre.set_compute_params(batch_size=batch_size)
        vre.run(out_dir, frames=frames, n_threads_data_storer=2, exception_mode="skip_representation")

    jsons = [json.load(open(next((out_dir/".logs").glob("*.json")), "r")) for out_dir in out_dirs]
    dfs = [pd.DataFrame(_json["run_stats"], index=frames).replace(1 <<31, float("nan")) for _json in jsons]
    avgs = pd.concat([df.mean() for df in dfs], axis=1)
    avgs.columns = [f"{bs=}" for bs in batch_sizes]
    for bs in batch_sizes[1:]:
        avgs[f"speedup-{bs}/{batch_sizes[0]}"] = avgs["bs=1"] / avgs[f"{bs=}"]
    print(avgs)
    avgs.to_csv("avgs.csv")

if __name__ == "__main__":
    main()
