#!/usr/bin/env python3
"""usage `vre_dir_analysis root_dir > res.json`. root_dir must be like: [subdir1/[repr1,...,reprn], subdir2[], ... ]"""
import pandas as pd
from pathlib import Path
import sys

def cnts(base_path: Path) -> pd.DataFrame:
    """compute the cunts of all the subdirs (vre out dirs) in a root dir. Max depth=2."""
    data = []
    for dir_path in base_path.iterdir(): # root_dir
        if not dir_path.is_dir():
            continue
        subdirs = [x for x in dir_path.iterdir() if x.is_dir()]
        if not {".logs", "rgb"}.issubset(map(lambda x: x.name, subdirs)): # not a vre dir, but something else
            continue

        for subdir in subdirs: # each vre_dir inside root_dir
            if subdir.name.startswith("."):
                continue
            npz_path = subdir / "npz"
            cnt = len(list(npz_path.iterdir())) if npz_path.exists() and npz_path.is_dir() else 0
            data.append([dir_path.name, subdir.name, cnt])

    df = pd.DataFrame(data, columns=["scene", "task", "counts"])
    f = lambda scene: pd.Series(dict(zip(scene["task"], scene["counts"]))).to_frame().T
    df2 = df.groupby("scene").apply(f).reset_index().drop(columns=["level_1"]).set_index("scene")
    df2 = df2.fillna(0).astype(int)
    df2 = df2[["rgb", *[c for c in df2.columns if c not in ("rgb", )]]]
    df2.loc["total"] = df2.sum()
    return df2

if __name__ == "__main__":
    res = cnts(Path(sys.argv[1]))
    print(res.T.to_json())
