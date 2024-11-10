"""compute or load statistics for a set of Normed Representations of a MultiTaskDataset"""
from __future__ import annotations
from typing import Tuple
from pathlib import Path
import os
import torch as tr
import numpy as np
from tqdm import trange

from vre.logger import vre_logger as logger
from vre.representations import NormedRepresentationMixin, Representation

TaskStatistics = Tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor] # (min, max, mean, std)

def load_external_statistics(reader: "MultiTaskDataset",
                             statistics: dict[str, TaskStatistics | list]) -> dict[str, TaskStatistics]:
    """loads statistics from an external source provided to the constructor"""
    name_to_task: dict[str, Representation] = reader.name_to_task
    tasks_no_classif = [t for t in set(reader.task_names) if isinstance(name_to_task[t], NormedRepresentationMixin)]
    assert (diff := set(tasks_no_classif).difference(statistics)) == set(), f"Missing tasks: {diff}"
    res: dict[str, TaskStatistics] = {}
    for k, v in statistics.items():
        if k in reader.task_names:
            res[k] = tuple(tr.Tensor(x) for x in v)
            assert all(_stat.shape == (nd := (name_to_task[k].n_channels, )) for _stat in res[k]), (res[k], nd)
    logger.info(f"External statistics provided: { {k: tuple(v[0].shape) for k, v in res.items()} }")
    return res

def compute_statistics(reader: "MultiTaskDataset") -> dict[str, TaskStatistics]:
    """computes statistics for all tasks that are not classification"""
    name_to_task: dict[str, Representation] = reader.name_to_task
    cache_path: Path = reader.path / ".task_statistics.npz"
    res: dict[str, TaskStatistics] = {}
    if reader.cache_task_stats and cache_path.exists():
        res = np.load(cache_path, allow_pickle=True)["arr_0"].item()
        logger.info(f"Loaded task statistics: { {k: tuple(v[0].shape) for k, v in res.items()} } from {cache_path}")
    missing_tasks = [t for t in set(reader.task_names).difference(res)
                     if isinstance(name_to_task[t], NormedRepresentationMixin)]
    if len(missing_tasks) == 0:
        return res
    logger.info(f"Computing global task statistics (dataset len {len(reader)}) for {missing_tasks}")
    res = {**res, **_compute_channel_level_stats(reader, missing_tasks)}
    logger.info(f"Computed task statistics: { {k: tuple(v[0].shape) for k, v in res.items()} }")
    np.savez(cache_path, res)
    return res

def _compute_channel_level_stats(reader: "MultiTaskDataset", missing_tasks: list[str]) -> dict[str, TaskStatistics]:
    # kinda based on: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def update(counts: tr.Tensor, counts_delta: float,  mean: tr.Tensor, m2: tr.Tensor,
                new_value: tr.Tensor) -> tuple[tr.Tensor, tr.Tensor, tr.Tensor]:
        new_count = counts + counts_delta
        batch_mean = new_value.nanmean(0)
        batch_var = ((new_value - batch_mean) ** 2).nansum(0)
        delta = batch_mean - mean
        new_count_no_zero = new_count + (new_count == 0) # add 1 (True) in case new_count is 0 to not divide by 0
        new_mean = mean + delta * counts_delta / new_count_no_zero
        new_m2 = m2 + batch_var + delta**2 * counts * counts_delta / new_count_no_zero
        assert not new_mean.isnan().any() and not new_m2.isnan().any(), (mean, new_mean, counts, counts_delta)
        return new_count, new_mean, new_m2

    name_to_task: dict[str, Representation] = reader.name_to_task
    assert all(mt := [isinstance(name_to_task[t], NormedRepresentationMixin) for t in missing_tasks]), mt
    assert len(missing_tasks) > 0, missing_tasks
    ch = {k: v[-1] if len(v) == 3 else 1 for k, v in reader.data_shape.items()}
    counts = {task_name: tr.zeros(ch[task_name]).long() for task_name in missing_tasks}
    mins = {task_name: tr.zeros(ch[task_name]).type(tr.float64) + 10**10 for task_name in missing_tasks}
    maxs = {task_name: tr.zeros(ch[task_name]).type(tr.float64) - 10**10 for task_name in missing_tasks}
    means_vec = {task_name: tr.zeros(ch[task_name]).type(tr.float64) for task_name in missing_tasks}
    m2s_vec = {task_name: tr.zeros(ch[task_name]).type(tr.float64) for task_name in missing_tasks}

    old_names, old_normalization = reader.task_names, reader.normalization
    reader.task_names, reader.normalization = missing_tasks, None # for reader[ix]
    res = {}
    bs = min(len(reader), reader.batch_size_stats)
    n = (len(reader) // bs) + (len(reader) % bs != 0)

    logger.debug(f"Global task statistics. Batch size: {bs}. N iterations: {n}.")
    for ix in trange(n, disable=os.getenv("STATS_PBAR", "0") == "0", desc="Computing stats"):
        item = reader[ix * bs: min(len(reader), (ix + 1) * bs)][0]
        for task in missing_tasks:
            item_flat_ch = item[task].reshape(-1, ch[task])
            item_no_nan = item_flat_ch.nan_to_num(0).type(tr.float64)
            mins[task] = tr.minimum(mins[task], item_no_nan.min(0)[0])
            maxs[task] = tr.maximum(maxs[task], item_no_nan.max(0)[0])
            counts_delta = (item_flat_ch == item_flat_ch).long().sum(0) # pylint: disable=comparison-with-itself
            counts[task], means_vec[task], m2s_vec[task] = \
                update(counts[task], counts_delta, means_vec[task], m2s_vec[task], item_no_nan)

    for task in missing_tasks:
        res[task] = (mins[task], maxs[task], means_vec[task], (m2s_vec[task] / counts[task]).sqrt())
        assert not any(x[0].isnan().any() for x in res[task]), (task, res[task])
    reader.task_names, reader.normalization = old_names, old_normalization
    return res
