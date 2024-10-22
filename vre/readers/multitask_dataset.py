#!/usr/bin/env python3
"""MultiTask Dataset module compatible with torch.utils.data.Dataset & DataLoader."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
from natsort import natsorted
from loggez import loggez_logger as logger
import torch as tr
import numpy as np
from torch.utils.data import Dataset
from tqdm import trange

from vre.stored_representation import StoredRepresentation

BuildDatasetTuple = Tuple[Dict[str, List[Path]], List[str]]
MultiTaskItem = Tuple[Dict[str, tr.Tensor], str, List[str]] # [{task: data}, stem(name) | list[stem(name)], [tasks]]
TaskStatistics = Tuple[tr.Tensor, tr.Tensor, tr.Tensor, tr.Tensor] # (min, max, mean, std)

class MultiTaskDataset(Dataset):
    """
    MultiTaskDataset implementation. Reads data from npz files and returns them as a dict.
    Parameters:
    - path: Path to the directory containing the npz files.
    - task_names: List of tasks that are present in the dataset. If set to None, will infer from the files on disk.
    - task_types: A dictionary of form {task_name: task_type} for the reader to call to read from disk, plot etc.
    - normalization: The normalization type used in __getitem__. Valid options are:
      - None: Reads the data as-is using task.read_from_disk(path)
      - 'min_max': Calls task.normalize(task.read_from_disk(path), mins[task], maxs[task])
      - 'standardization': Calls task.standardize(task.read_from_disk(path), means[task], stds[task])
      If normalization is not 'none', then task-level statistics will be computed. Environmental variable
      STATS_PBAR=0/1 enables tqdm during statistics computation.
    - handle_missing_data: Modes to handle missing data. Valid options are:
      - 'drop': Drop the data point if any of the representations is missing.
      - 'fill_{none,zero,nan}': Fill the missing data with Nones, zeros or NaNs.
    - files_suffix: What suffix to look for when creating the dataset. Valid values: 'npy' or 'npz'.
    - cache_task_stats: If set to True, the statistics will be cached at '{path}/.task_statistics.npz'. Can be enabled
    using the environmental variable STATS_CACHE=1. Defaults to False.
    - batch_size_stats: Controls the batch size during statistics computation. Can be enabled by environmental variable
    STATS_BATCH_SIZE. Defaults to 1.

    Expected directory structure:
    path/
    - task_1/0.npz, ..., N.npz
    - ...
    - task_n/0.npz, ..., N.npz

    Names can be in a different format (i.e. 2022-01-01.npz), but must be consistent and equal across all tasks.
    """

    def __init__(self, path: Path,
                 task_names: list[str],
                 task_types: dict[str, type],
                 normalization: str | None | dict[str],
                 handle_missing_data: str = "fill_none",
                 files_suffix: str = "npz",
                 cache_task_stats: bool = (os.getenv("STATS_CACHE", "0") == "1"),
                 batch_size_stats: int = int(os.getenv("STATS_BATCH_SIZE", "1")),
                 statistics: dict[str, TaskStatistics] | None = None,
    ):
        assert Path(path).exists(), f"Provided path '{path}' doesn't exist!"
        assert handle_missing_data in ("drop", "fill_none", "fill_zero", "fill_nan"), \
            f"Invalid handle_missing_data mode: {handle_missing_data}"
        assert isinstance(task_names, Iterable), type(task_names)
        self.path = Path(path).absolute()
        self.handle_missing_data = handle_missing_data
        self.suffix = files_suffix
        self.files_per_repr, self.file_names = self._build_dataset(task_types, task_names) # + handle_missing_data
        self.cache_task_stats = cache_task_stats
        self.batch_size_stats = batch_size_stats

        assert all(isinstance(x, str) for x in task_names), tuple(zip(task_names, (type(x) for x in task_names)))
        assert (diff := set(self.files_per_repr).difference(task_names)) == set(), f"Not all tasks in files: {diff}"
        self.task_types = {k: v for k, v in task_types.items() if k in task_names} # all task_types must be provided!
        self.task_names = sorted(task_names)
        logger.info(f"Tasks used in this dataset: {self.task_names}")

        if normalization is not None:
            if isinstance(normalization, str):
                logger.info(f"Normalization provided as a string ({normalization}). Setting all tasks to this")
                normalization: dict[str, str] = {task: normalization for task in self.task_names}
            if "*" in normalization.keys(): # for the lazy, we can put {"*": "standardization", "depth": "min_max"}
                value = normalization.pop("*")
                for missing_task in set(self.task_names).difference(normalization.keys()):
                    normalization[missing_task] = value
            assert all(n in ("min_max", "standardization") for n in normalization.values()), normalization
            assert all(k in task_names for k in normalization.keys()), set(normalization).difference(task_names)
        self.normalization: dict[str, str] | None = normalization

        self._data_shape: tuple[int, ...] | None = None
        self._tasks: list[StoredRepresentation] | None = None
        self._default_vals: dict[str, tr.Tensor] | None = None
        if statistics is not None:
            self._statistics = self._load_external_statistics(statistics)
        else:
            self._statistics = None if normalization is None else self._compute_statistics()
        if self._statistics is not None:
            for task_name, task in self.name_to_task.items():
                if not task.is_classification:
                    task.set_normalization(self.normalization[task_name], self._statistics[task_name])

    # Public methods and properties

    @property
    def name_to_task(self) -> dict[str, StoredRepresentation]:
        """A dict that maps the name of the task to the task"""
        return {task.name: task for task in self.tasks}

    @property
    def default_vals(self) -> dict[str, tr.Tensor]:
        """default values for __getitem__ if item is not on disk but we retrieve a full batch anyway"""
        _default_val = float("nan") if self.handle_missing_data == "fill_nan" else 0
        return {task: None if self.handle_missing_data == "fill_none" else tr.full(self.data_shape[task], _default_val)
                for task in self.task_names}

    @property
    def data_shape(self) -> dict[str, tuple[int, ...]]:
        """Returns a {task: shape_tuple} for all representations. At least one npz file must exist for each."""
        first_npz = {task: [_v for _v in files if _v is not None][0] for task, files in self.files_per_repr.items()}
        data_shape = {task: self.name_to_task[task].load_from_disk(first_npz[task]).shape for task in self.task_names}
        return {task: tuple(shape) for task, shape in data_shape.items()}

    @property
    def mins(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[0] for k, v in self._statistics.items() if k in self.task_names}

    @property
    def maxs(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[1] for k, v in self._statistics.items() if k in self.task_names}

    @property
    def means(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[2] for k, v in self._statistics.items() if k in self.task_names}

    @property
    def stds(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[3] for k, v in self._statistics.items() if k in self.task_names}

    @property
    def tasks(self) -> list[StoredRepresentation]:
        """
        Returns a list of instantiated tasks in the same order as self.task_names. Overwrite this to add
        new tasks and semantics (i.e. plot_fn or doing some preprocessing after loading from disk in some tasks.
        """
        if self._tasks is None:
            self._tasks = []
            for task_name in self.task_names:
                t = self.task_types[task_name]
                try:
                    t = t(task_name) # hack for not isinstance(self.task_types, StoredRepresentation) but callable
                except Exception:
                    pass
                self._tasks.append(t)
            assert all(t.name == t_n for t, t_n in zip(self._tasks, self.task_names)), (self.task_names, self._tasks)
        return self._tasks

    def collate_fn(self, items: list[MultiTaskItem]) -> MultiTaskItem:
        """
        given a list of items (i.e. from a reader[n:n+k] call), return the item batched on 1st dimension.
        Nones (missing data points) are turned into nans as per the data shape of that dim.
        """
        assert all(item[2] == self.task_names for item in items), ([item[2] for item in items], self.task_names)
        items_name = [item[1] for item in items]
        res = {k: tr.zeros(len(items), *self.data_shape[k]).float() for k in self.task_names} # float32 always
        for i in range(len(items)):
            for k in self.task_names:
                try:
                    res[k][i][:] = items[i][0][k] if items[i][0][k] is not None else float("nan")
                except Exception as e:
                    print(k, items)
                    raise e
        return res, items_name, self.task_names

    def add_task(self, task: StoredRepresentation, overwrite: bool=False):
        """Safely adds a task to this reader. Most likely can be optimized"""
        logger.debug(f"Adding a new task: '{task.name}'")
        if task.name in self.task_names:
            if overwrite:
                self.remove_task(task.name)
            else:
                raise ValueError(f"Task '{task.name}' already exists: {self.task_names}")
        self.task_names = sorted([*self.task_names, task.name])
        self.task_types[task.name] = task
        self._tasks = None
        self.files_per_repr, self.file_names = self._build_dataset(self.task_types, self.task_names)

    def remove_task(self, task_name: str):
        """Safely removes a task from this reader"""
        logger.debug(f"Removing a task: '{task_name}'")
        assert task_name in self.task_names, f"Task '{task_name}' doesn't exist: {self.task_names}"
        self.task_names = sorted(name for name in self.task_names if name != task_name)
        del self.task_types[task_name]
        self._tasks = None
        self.files_per_repr, self.file_names = self._build_dataset(self.task_types, self.task_names)

    # Private methods

    def _get_all_npz_files(self) -> dict[str, list[Path]]:
        """returns a dict of form: {"rgb": ["0.npz", "1.npz", ..., "N.npz"]}"""
        assert self.suffix == "npz", f"Only npz supported right now (though trivial to update): {self.suffix}"
        in_files = {}
        all_repr_dirs: list[str] = [x.name for x in self.path.iterdir() if x.is_dir()]
        for repr_dir_name in all_repr_dirs:
            dir_name = self.path / repr_dir_name
            if all(f.is_dir() for f in dir_name.iterdir()): # dataset is stored as repr/part_x/0.npz, ..., part_k/n.npz
                all_files = []
                for part in dir_name.iterdir():
                    all_files.extend(part.glob(f"*.{self.suffix}"))
            else: # dataset is stored as repr/0.npz, ..., repr/n.npz
                all_files = dir_name.glob(f"*.{self.suffix}")
            in_files[repr_dir_name] = natsorted(all_files, key=lambda x: x.name) # important: use natsorted() here
        assert not any(len(x) == 0 for x in in_files.values()), f"{ [k for k, v in in_files.items() if len(v) == 0] }"
        return in_files

    def _build_dataset(self, task_types: dict[str, StoredRepresentation], task_names: list[str]) -> BuildDatasetTuple:
        logger.debug(f"Building dataset from: '{self.path}'")
        all_npz_files = self._get_all_npz_files()
        all_files: dict[str, dict[str, Path]] = {k: {_v.name: _v for _v in v} for k, v in all_npz_files.items()}

        relevant_tasks_for_files = set() # hsv requires only rgb, so we look at dependencies later on
        for task_name in task_names:
            relevant_tasks_for_files.update(task_types[task_name].dep_names)
        if (diff := relevant_tasks_for_files.difference(all_files)) != set():
            raise FileNotFoundError(f"Missing files for {diff}.\nFound on disk: {[*all_files]}")
        names_to_tasks: dict[str, list[str]] = {} # {name: [task]}
        for task_name in relevant_tasks_for_files: # just the relevant tasks
            for path_name in all_files[task_name].keys():
                names_to_tasks.setdefault(path_name, [])
                names_to_tasks[path_name].append(task_name)

        if self.handle_missing_data == "drop":
            b4 = len(names_to_tasks)
            names_to_tasks = {k: v for k, v in names_to_tasks.items() if len(v) == len(relevant_tasks_for_files)}
            logger.debug(f"Dropped {b4 - len(names_to_tasks)} files not in all tasks")
        all_names: list[str] = natsorted(names_to_tasks.keys())
        logger.info(f"Total files: {len(names_to_tasks)} per task across {len(task_names)} tasks")

        files_per_task: dict[str, list[str | None] | list[list[str] | None]] = {task: [] for task in task_names}
        for name in all_names:
            for task in task_names:
                all_deps_exist = set(deps := task_types[task].dep_names).issubset(names_to_tasks[name])
                if not all_deps_exist:
                    files_per_task[task].append(None) # if any of the deps don't exist for this task, skip it.
                else:
                    paths = [all_files[dep][name] for dep in deps]
                    files_per_task[task].append(paths if len(deps) > 1 else paths[0])
        return files_per_task, all_names

    def _compute_statistics(self) -> dict[str, TaskStatistics]:
        cache_path = self.path / ".task_statistics.npz"
        res: dict[str, TaskStatistics] = {}
        if self.cache_task_stats and cache_path.exists():
            res = np.load(cache_path, allow_pickle=True)["arr_0"].item()
            logger.info(f"Loaded task statistics: { {k: tuple(v[0].shape) for k, v in res.items()} } from {cache_path}")
        missing_tasks = [t for t in set(self.task_names).difference(res) if not self.name_to_task[t].is_classification]
        if len(missing_tasks) == 0:
            return res
        logger.info(f"Computing global task statistics (dataset len {len(self)}) for {missing_tasks}")
        res = {**res, **self._compute_channel_level_stats(missing_tasks)}
        logger.info(f"Computed task statistics: { {k: tuple(v[0].shape) for k, v in res.items()} }")
        np.savez(cache_path, res)
        return res

    def _compute_channel_level_stats(self, missing_tasks: list[str]) -> dict[str, TaskStatistics]:
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

        assert not any(mt := [self.name_to_task[t].is_classification for t in missing_tasks]), mt
        assert len(missing_tasks) > 0, missing_tasks
        ch = {k: v[-1] if len(v) == 3 else 1 for k, v in self.data_shape.items()}
        counts = {task_name: tr.zeros(ch[task_name]).long() for task_name in missing_tasks}
        mins = {task_name: tr.zeros(ch[task_name]).type(tr.float64) + 10**10 for task_name in missing_tasks}
        maxs = {task_name: tr.zeros(ch[task_name]).type(tr.float64) - 10**10 for task_name in missing_tasks}
        means_vec = {task_name: tr.zeros(ch[task_name]).type(tr.float64) for task_name in missing_tasks}
        m2s_vec = {task_name: tr.zeros(ch[task_name]).type(tr.float64) for task_name in missing_tasks}

        old_names, old_normalization = self.task_names, self.normalization
        self.task_names, self.normalization = missing_tasks, None # for self[ix]
        res = {}
        bs = min(len(self), self.batch_size_stats)
        n = (len(self) // bs) + (len(self) % bs != 0)

        logger.debug(f"Global task statistics. Batch size: {bs}. N iterations: {n}.")
        for ix in trange(n, disable=os.getenv("STATS_PBAR", "0") == "0", desc="Computing stats"):
            item = self[ix * bs: min(len(self), (ix + 1) * bs)][0]
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
        self.task_names, self.normalization = old_names, old_normalization
        return res

    def _load_external_statistics(self, statistics: dict[str, TaskStatistics | list]) -> dict[str, TaskStatistics]:
        tasks_no_classif = [t for t in set(self.task_names) if not self.name_to_task[t].is_classification]
        assert (diff := set(tasks_no_classif).difference(statistics)) == set(), f"Missing tasks: {diff}"
        res: dict[str, TaskStatistics] = {}
        for k, v in statistics.items():
            if k in self.task_names:
                res[k] = tuple(tr.Tensor(x) for x in v)
                assert all(_stat.shape == (nd := (self.name_to_task[k].n_channels, )) for _stat in res[k]), (res[k], nd)
        logger.info(f"External statistics provided: { {k: tuple(v[0].shape) for k, v in res.items()} }")
        return res

    # Python magic methods (pretty printing the reader object, reader[0], len(reader) etc.)

    def __getitem__(self, index: int | str | slice | list[int, str] | tuple[int, str]) -> MultiTaskItem:
        """Read the data all the desired nodes"""
        assert isinstance(index, (int, slice, list, tuple, str)), type(index)
        if isinstance(index, slice):
            assert index.start is not None and index.stop is not None and index.step is None, "Only reader[l:r] allowed"
            index = list(range(index.stop)[index])
        if isinstance(index, (list, tuple)):
            return self.collate_fn([self.__getitem__(ix) for ix in index])
        if isinstance(index, str):
            return self.__getitem__(self.file_names.index(index))

        res: dict[str, tr.Tensor] = {}
        for task_name in self.task_names:
            task = [t for t in self.tasks if t.name == task_name][0]
            file_path = self.files_per_repr[task_name][index]
            res[task_name] = self.default_vals[task_name] if file_path is None else task.load_from_disk(file_path)
            if not task.is_classification:
                if self.normalization is not None and self.normalization[task_name] == "min_max":
                    res[task_name] = task.normalize(res[task_name])
                if self.normalization is not None and self.normalization[task_name] == "standardization":
                    res[task_name] = task.standardize(res[task_name])
        return (res, self.file_names[index], self.task_names)

    def __len__(self) -> int:
        return len(self.files_per_repr[self.task_names[0]]) # all of them have the same number (filled with None or not)

    def __str__(self):
        f_str = f"[{str(type(self)).rsplit('.', maxsplit=1)[-1][0:-2]}]"
        f_str += f"\n - Path: '{self.path}'"
        f_str += f"\n - Tasks ({len(self.tasks)}): {self.tasks}"
        f_str += f"\n - Length: {len(self)}"
        f_str += f"\n - Handle missing data mode: '{self.handle_missing_data}'"
        f_str += f"\n - Normalization: '{self.normalization}'"
        return f_str

    def __repr__(self):
        return str(self)
