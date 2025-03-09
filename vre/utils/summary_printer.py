"""summary printer at the end of a vre run"""
import numpy as np
from .utils import str_maxk

class SummaryPrinter:
    """summary printer at the end of a vre run"""
    def __init__(self, representations: list[str], runtime_args: "VRERuntimeArgs"):
        self.representations = representations
        self.repr_metadatas: dict[str, "RepresentationMetadata" | None] = {r: None for r in representations}
        self.runtime_args = runtime_args.to_dict()

    @property
    def run_stats(self) -> dict[str, dict[str, float]]:
        """The run stats of each representation that was registered to this run or older ones (only computed values)"""
        res = {}
        for r in self.repr_metadatas.values():
            if r is not None:
                _res = {k: v for k, v in r.run_stats.items() if v is not None}
                if len(_res) > 0:
                    res[r.repr_name] = _res
        return res

    def __call__(self) -> str:
        """returns a pretty formatted string of the metadata"""
        # vre_run_stats_np = np.array([list(x.values()) for x in self.run_stats.values()]).T.round(3)
        # vre_run_stats_np[abs(vre_run_stats_np - float(1<<31)) < 1e-2] = float("nan")
        res = ""
        frames = self.runtime_args["frames"]
        chosen_frames = sorted(np.random.choice(frames, size=min(5, len(frames)), replace=False))
        res = f"{'Name':<20}" + "|" + "|".join([f"{str_maxk(k, 9):<9}" for k in map(str, chosen_frames)]) + "|Total"
        for vrepr in self.representations:
            if vrepr not in self.run_stats:
                continue
            stats = np.array(list(self.run_stats[vrepr].values())).round(3)
            stats[abs(stats - float(1<<31)) < 1e-2] = float("nan")
            res += "\n" + f"{str_maxk(vrepr, 20):<20}"
            for chosen_frame in chosen_frames:
                res += "|" + f"{stats[frames.index(chosen_frame)]:<9}"
            res += "|" + f"{stats.sum().round(2)}"
        return res
