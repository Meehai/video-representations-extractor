"""DataStorer module -- Multi-threaded wrapper for DataWriter"""
from threading import Thread
from multiprocessing import cpu_count
from queue import Queue
from time import time
import numpy as np

from .representations import ReprOut
from .data_writer import DataWriter
from .logger import vre_logger as logger

class DataStorer:
    """
    Equivalent of DataLoader on top of a Dataset -> DataStorer on top of a DataWriter
    Parameters:
    - data_writer The DataWriter object used by this DataStorer (akin to Dataset and DataLoader)
    - n_threads_data_storer The number of workers used for the ThreadPool that stores data at each step. This is
    needed because storing data takes a lot of time sometimes, even more than the computation itself.
    """
    def __init__(self, data_writer: DataWriter, n_threads: int):
        assert n_threads >= 0, n_threads
        self.data_writer = data_writer
        self.n_threads = min(n_threads, cpu_count())
        self.threads: list[Thread] = []
        if n_threads >= 1:
            self.queue: Queue = Queue(maxsize=1) # maxisze=1 because we use many shared memory threads.
            for _ in range(n_threads):
                self.threads.append(thr := Thread(target=self._worker_fn, daemon=True, args=(self.queue, )))
                thr.start()
        logger.debug(f"[DataStorer] Set up with {n_threads} threads.")

    def _worker_fn(self, queue: Queue):
        while True: # Since these threads are daemon, they die when main thread dies. We could use None or smth to break
            self.data_writer(*queue.get())
            queue.task_done()

    def join_with_timeout(self, timeout: int):
        """calls queue.join() but throws after timeout seconds if it doesn't end"""
        if self.n_threads == 0:
            return
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

    def __call__(self, name: str, y_repr: ReprOut, imgs: np.ndarray | None, l: int, r: int):
        assert isinstance(y_repr, ReprOut), f"{name=}, {type(y_repr)=}"
        if self.n_threads == 0:
            self.data_writer(name, y_repr, imgs, l, r)
        else:
            self.queue.put((name, y_repr, imgs, l, r), block=True, timeout=30)

    def __repr__(self):
        return f"""[DataStorer]
- Num threads: {self.n_threads} (0 = using main thread)
{self.data_writer}"""
