#!/usr/bin/env python3
"""MultiTask Dataset module compatible with torch.utils.data.Dataset & DataLoader."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Union
from copy import deepcopy
from natsort import natsorted
import torch as tr
from torch.utils.data import Dataset

from vre.representations import Representation, NormedRepresentationMixin, IORepresentationMixin, TaskMapper
from vre.logger import vre_logger as logger

from .statistics import compute_statistics, load_external_statistics, TaskStatistics

BuildDatasetTuple = Tuple[Dict[str, List[Path]], List[str]]
MultiTaskItem = Tuple[Dict[str, tr.Tensor], str, List[str]] # [{task: data}, stem(name) | list[stem(name)], [tasks]]
Repr = Union[Representation | IORepresentationMixin]

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
    - handle_missing_data: Modes to handle missing data. Defaults to 'fill_none'. Valid options are:
      - 'raise': Raise exception if any missing data.
      - 'drop': Drop the data point if any of the representations is missing.
      - 'fill_{none,zero,nan}': Fill the missing data with Nones, zeros or NaNs.
    - files_suffix: What suffix to look for when creating the dataset. Valid values: 'npy' or 'npz'.
    - cache_task_stats: If set to True, the statistics will be cached at '{path}/.task_statistics.npz'. Can be enabled
    using the environmental variable STATS_CACHE=1. Defaults to False.
    - batch_size_stats: Controls the batch size during statistics computation. Can be enabled by environmental variable
    STATS_BATCH_SIZE. Defaults to 1.
    - statistics The dictionary of statistics which can be externally provided too, otherwise computes or loads them
    from the datasert dir if normalization is set.

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
                 cache_task_stats: bool = (os.getenv("STATS_CACHE", "1") == "1"),
                 batch_size_stats: int = int(os.getenv("STATS_BATCH_SIZE", "1")),
                 statistics: dict[str, TaskStatistics] | None = None,
    ):
        assert Path(path).exists(), f"Provided path '{path}' doesn't exist!"
        assert handle_missing_data in ("drop", "fill_none", "fill_zero", "fill_nan", "raise"), \
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
        # deepcopy is needed so we don't overwrite the properties (i.e. normalization) of the global task types.
        self.task_types = {k: deepcopy(v) for k, v in task_types.items() if k in task_names}
        self.task_names = sorted(task_names)
        logger.info(f"Tasks used in this dataset: {self.task_names}")

        if normalization is not None:
            if isinstance(normalization, str):
                logger.debug(f"Normalization provided as a string ({normalization}). Setting all tasks to this")
                normalization: dict[str, str] = {task: normalization for task in self.task_names}
            if "*" in normalization.keys(): # for the lazy, we can put {"*": "standardization", "depth": "min_max"}
                value = normalization.pop("*")
                for missing_task in set(self.task_names).difference(normalization.keys()):
                    normalization[missing_task] = value
            assert all(n in ("min_max", "standardization") for n in normalization.values()), normalization
            assert all(k in task_names for k in normalization.keys()), set(normalization).difference(task_names)
        self.normalization: dict[str, str] | None = normalization

        self._data_shape: tuple[int, ...] | None = None
        self._tasks: list[Repr] | None = None
        self._default_vals: dict[str, tr.Tensor] | None = None
        self._statistics: dict[str, TaskStatistics] | None = None
        if statistics is not None:
            self._statistics = load_external_statistics(self, statistics)

    # Public methods and properties

    @property
    def statistics(self) -> dict[str, TaskStatistics] | None:
        """returns the statistics of this dataset for all NormedRepresentation tasks, None otherwise"""
        if self.normalization is None:
            return None
        if self._statistics is None:
            self._statistics = compute_statistics(self)
        for task_name, task in self.name_to_task.items():
            if isinstance(task, NormedRepresentationMixin) and task.normalization is None:
                task.set_normalization(self.normalization[task_name], self._statistics[task_name])
        return self._statistics

    @property
    def name_to_task(self) -> dict[str, Repr]:
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
        data_shape = {}
        for task_name, task in self.name_to_task.items():
            if isinstance(task, TaskMapper) and task.dependencies[0] != task:
                data_shape[task_name] = task.compute_from_dependencies_paths(first_npz[task_name]).shape
            else:
                data_shape[task_name] = task.disk_to_memory_fmt(task.load_from_disk(first_npz[task_name])).shape
        return data_shape

    @property
    def mins(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[0] for k, v in self.statistics.items() if k in self.task_names}

    @property
    def maxs(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[1] for k, v in self.statistics.items() if k in self.task_names}

    @property
    def means(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[2] for k, v in self.statistics.items() if k in self.task_names}

    @property
    def stds(self) -> dict[str, tr.Tensor]:
        """returns a dict {task: mins[task]} for all the tasks if self.statistics exists"""
        assert self.normalization is not None, "No statistics for normalization is None"
        return {k: v[3] for k, v in self.statistics.items() if k in self.task_names}

    @property
    def tasks(self) -> list[Repr]:
        """
        Returns a list of instantiated tasks in the same order as self.task_names. Overwrite this to add
        new tasks and semantics (i.e. plot_fn or doing some preprocessing after loading from disk in some tasks.
        """
        if self._tasks is None:
            self._tasks = []
            for task_name in self.task_names:
                self._tasks.append(self.task_types[task_name])
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

    def add_task(self, task: Repr, overwrite: bool=False):
        """Safely adds a task to this reader. Most likely can be optimized"""
        logger.info(f"Adding a new task: '{task.name}'")
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
        logger.info(f"Removing a task: '{task_name}'")
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
        all_repr_dirs: list[str] = [x.name for x in self.path.iterdir() if x.is_dir() and not x.name.startswith(".")]
        for repr_dir_name in all_repr_dirs:
            dir_name = self.path / repr_dir_name
            if all(f.is_dir() for f in dir_name.iterdir()): # dataset is stored as repr/part_x/0.npz, ..., part_k/n.npz
                all_files = []
                for part in dir_name.iterdir():
                    all_files.extend(part.glob(f"*.{self.suffix}"))
            else: # dataset is stored as repr/0.npz, ..., repr/n.npz
                all_files = dir_name.glob(f"*.{self.suffix}")
            all_files = [x for x in all_files if not x.name.endswith("_extra.npz")] # important: remove xxx_extra.npz
            in_files[repr_dir_name] = natsorted(all_files, key=lambda x: x.name) # important: use natsorted() here
        assert not any(len(x) == 0 for x in in_files.values()), f"{ [k for k, v in in_files.items() if len(v) == 0] }"
        return in_files

    def _build_dataset(self, task_types: dict[str, Repr], task_names: list[str]) -> BuildDatasetTuple:
        logger.debug(f"Building dataset from: '{self.path}'")
        all_npz_files = self._get_all_npz_files()
        all_files: dict[str, dict[str, Path]] = {k: {_v.name: _v for _v in v} for k, v in all_npz_files.items()}

        if (diff := set(task_names).difference(all_files)) != set():
            logger.debug(f"The following tasks do not have data on disk: {list(diff)}. Checking dependencies.")
        relevant_tasks_for_files = set() # hsv requires only rgb, so we look at dependencies later on
        for task_name in task_names:
            assert task_name in task_types, f"Task '{task_name}' not in {task_types=}. Check your tasks."
            if task_name not in diff and task_types[task_name].dep_names != [task_name]:
                logger.debug(f"Upating the deps of '{task_name}' as all its data is on disk!")
                task_types[task_name].dependencies = [task_types[task_name]]
            relevant_tasks_for_files.update(task_types[task_name].dep_names)
        if (diff := relevant_tasks_for_files.difference(all_files)) != set():
            raise FileNotFoundError(f"Missing files for {diff}.\nFound on disk: {[*all_files]}")

        names_to_tasks: dict[str, list[str]] = {} # {name: [task]}
        for task_name in relevant_tasks_for_files: # just the relevant tasks
            for path_name in all_files[task_name].keys():
                names_to_tasks.setdefault(path_name, [])
                names_to_tasks[path_name].append(task_name)

        if self.handle_missing_data == "raise":
            for k, _v in names_to_tasks.items():
                if (v := set(_v)) != (first := set(names_to_tasks[next(iter(names_to_tasks))])):
                    raise ValueError(f"Key '{k}' has different files:\n-{v=}\n-{first=}\n-{v-first=}\n-{first-v=}")

        if self.handle_missing_data == "drop":
            b4 = len(names_to_tasks)
            names_to_tasks = {k: v for k, v in names_to_tasks.items() if len(v) == len(relevant_tasks_for_files)}
            logger.debug(f"Dropped {b4 - len(names_to_tasks)} files not in all tasks")
        all_names: list[str] = natsorted(names_to_tasks.keys())
        logger.debug(f"Total files: {len(names_to_tasks)} per task across {len(task_names)} tasks")

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

    def _get_one_item(self, index: int) -> MultiTaskItem:
        assert isinstance(index, int), type(index)
        res: dict[str, tr.Tensor] = {}
        for task_name in self.task_names:
            task = [t for t in self.tasks if t.name == task_name][0]
            file_path = self.files_per_repr[task_name][index]
            if file_path is None:
                res[task_name] = self.default_vals[task_name]
            else:
                if isinstance(task, TaskMapper) and task.dependencies[0] != task:
                    np_memory_data = task.compute_from_dependencies_paths(file_path)
                else: # can also be TaskMapper here too, but with deps[0] == task (pre-computed)
                    np_memory_data = task.disk_to_memory_fmt(task.load_from_disk(file_path))

                if isinstance(task, NormedRepresentationMixin) and self.statistics is not None:
                    np_memory_data = task.normalize(np_memory_data)
                res[task_name] = tr.from_numpy(np_memory_data)
        # TODO: why is self.task_names require here. It's already in res.keys().
        return (res, self.file_names[index], self.task_names)

    # Python magic methods (pretty printing the reader object, reader[0], len(reader) etc.)
    def __getitem__(self, index: int | str | slice | list[int, str] | tuple[int, str]) -> MultiTaskItem:
        """Read the data all the desired nodes"""
        assert isinstance(index, (int, slice, list, tuple, str)), type(index)
        if isinstance(index, slice):
            assert index.start is not None and index.stop is not None and index.step is None, "Only reader[l:r] allowed"
            index = list(range(index.start, index.stop))
        if isinstance(index, (list, tuple)):
            return self.collate_fn([self.__getitem__(ix) for ix in index])
        if isinstance(index, str):
            return self.__getitem__(self.file_names.index(index))
        return self._get_one_item(index)

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
