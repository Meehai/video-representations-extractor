"""AtomicOpen -- copy pasta from https://stackoverflow.com/questions/489861/locking-a-file-in-python"""
from io import FileIO
import os


try:
    # Posix based file locking (Linux, Ubuntu, MacOS, etc.)
    import fcntl

    def lock_file(f: FileIO):
        """locks a file"""
        if f.writable():
            fcntl.flock(f, fcntl.LOCK_EX)

    def unlock_file(f: FileIO):
        """unlocks a file"""
        if f.writable():
            fcntl.flock(f, fcntl.LOCK_UN)

except ModuleNotFoundError:
    # Windows file locking
    import msvcrt
    def file_size(f: FileIO):
        """file size"""
        return os.path.getsize(os.path.realpath(f.name))

    def lock_file(f: FileIO):
        """locks a file"""
        msvcrt.locking(f.fileno(), msvcrt.LK_RLCK, file_size(f))

    def unlock_file(f: FileIO):
        """unlocks a file"""
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_size(f))

class AtomicOpen:
    """
    lass for ensuring that all file operations are atomic, treat initialization like a standard call to 'open' that
    happens to be atomic. This file opener *must* be used in a "with" block.
    Open the file with arguments provided by user. Then acquire a lock on that file object.
    """
    def __init__(self, path: str, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.file: FileIO = open(path, *args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs):
        return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        os.fsync(self.file.fileno())
        # Release the lock on the file.
        unlock_file(self.file)
        self.file.close()
        # Handle exceptions that may have come up during execution, by default any exceptions are raised to the user.
        return exc_type is None
