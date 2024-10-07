"""DataStorer module -- used to store npy/png files in a multi threaded way if needed"""
from threading import Thread
from multiprocessing import cpu_count
from queue import Queue
from time import time
import numpy as np

from .vre_runtime_args import VRERuntimeArgs
from .utils import image_write, RepresentationOutput
from .logger import vre_logger as logger

class DataStorer:
    """Class used to store the representations on disk. It supports multi-threading so we don't block the compute one"""
    def __init__(self, n_threads: int):
        assert n_threads >= 0, n_threads
        self.n_threads = min(n_threads, cpu_count())
        self.queue: Queue = Queue(maxsize=1) # maxisze=1 because we use many shared memory threads.
        self.threads: list[Thread] = []
        for _ in range(n_threads):
            self.threads.append(thr := Thread(target=DataStorer._worker_fn, daemon=True, args=(self.queue, )))
            thr.start()
        logger.debug(f"[DataStorer] Set up with {n_threads} threads.")

    @staticmethod
    def _worker_fn(queue: Queue):
        while True: # Since these threads are daemon, they die when main thread dies. We could use None or smth to break
            DataStorer._store_data(*queue.get())
            queue.task_done()

    @staticmethod
    def _store_data(name: str, y_repr: RepresentationOutput, imgs: np.ndarray | None,
                    l: int, r: int, runtime_args: VRERuntimeArgs, frame_size: tuple[int, int]):
        """store the data in the right format"""
        if runtime_args.export_png:
            if (o_s := runtime_args.output_sizes[name]) != "native": # if native, godbless on the expected sizes.
                h, w = frame_size if o_s == "video_shape" else o_s
                assert imgs.shape == (r - l, h, w, 3), (imgs.shape, (r - l, h, w, 3))
            assert imgs is not None
            assert imgs.dtype == np.uint8, imgs.dtype

        for i, t in enumerate(range(l, r)):
            if runtime_args.export_npy:
                if not runtime_args.npy_paths[name][t].exists():
                    np.savez(runtime_args.npy_paths[name][t], y_repr.output[i])
                    if (extra := y_repr.extra) is not None and len(y_repr.extra) > 0:
                        assert len(extra) == r - l, f"Extra must be a list of len ({len(extra)}) = batch_size ({r-l})"
                        np.savez(runtime_args.npy_paths[name][t].parent / f"{t}_extra.npz", extra[i])
            if runtime_args.export_png:
                if not runtime_args.png_paths[name][t].exists():
                    image_write(imgs[i], runtime_args.png_paths[name][t])

    def join_with_timeout(self, timeout: int):
        """calls queue.join() but throws after timeout seconds if it doesn't end"""
        logger.debug(f"Waiting for {self.queue.unfinished_tasks} leftover enqueued tasks")
        self.queue.all_tasks_done.acquire()
        try:
            endtime = time() + timeout
            while self.queue.unfinished_tasks:
                remaining = endtime - time()
                if remaining <= 0.0:
                    raise RuntimeError("Queue has not finished the join() before timeout")
                self.queue.all_tasks_done.wait(remaining)
        finally:
            self.queue.all_tasks_done.release()

    def __call__(self, name: str, y_repr: RepresentationOutput, imgs: np.ndarray | None,
                 l: int, r: int, runtime_args: VRERuntimeArgs, frame_size: tuple[int, int]):
        assert isinstance(y_repr, RepresentationOutput), f"{name=}, {type(y_repr)=}"
        self.queue.put((name, y_repr, imgs, l, r, runtime_args, frame_size), block=True, timeout=30)
        if self.n_threads == 0: # call here if no threads are used.
            self._store_data(*self.queue.get())
            self.queue.task_done()
