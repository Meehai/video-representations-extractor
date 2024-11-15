"""DataStorer module -- Multi-threaded wrapper for DataWriter"""
from threading import Thread
from multiprocessing import cpu_count
from queue import Queue, Empty
from copy import deepcopy
from time import time

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
                self.threads.append(thr := Thread(target=self._worker_fn, daemon=True))
                thr.start()
        logger.debug(f"[{self.data_writer.rep.name}] Set up with {n_threads} threads.")

    def _worker_fn(self):
        while True: # This loop is ended via `self.join_with_timeout` or `__del__` or manually setting queue to None.
            if self.queue is None:
                logger.debug(f"[{self.data_writer.rep.name}] Queue of is None. Closing.")
                break
            try:
                args = self.queue.get(timeout=1)
            except (Empty, AttributeError): # AttributeError if it became 'None' during the if and the .get()
                continue
            self.data_writer(*args)
            self.queue.task_done()

    def join_with_timeout(self, timeout: int):
        """calls queue.join() but throws after timeout seconds if it doesn't end"""
        if self.n_threads == 0:
            return
        logger.debug(f"[{self.data_writer.rep.name}] Waiting for {self.queue.unfinished_tasks} "
                     "leftover enqueued tasks")
        assert self.queue is not None, "Queue was closed, create a new DataStorer object..."
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
        self.queue = None

    def __call__(self, y_repr: ReprOut):
        assert isinstance(y_repr, ReprOut), f"{self.data_writer.rep=}, {type(y_repr)=}"
        if self.n_threads == 0:
            self.data_writer(y_repr)
        else:
            assert self.queue is not None, "Queue was closed, create a new DataStorer object..."
            self.queue.put((deepcopy(y_repr), ), block=True, timeout=30)

    def __repr__(self):
        return f"""[DataStorer]
- Num threads: {self.n_threads} (0 = only using main thread)
{self.data_writer}"""

    def __del__(self):
        self.queue = None
