import contextlib
import os


@contextlib.contextmanager
def suppress_fd_stderr():
    """
    Open JTalk が stderr fd に直接書く出力を抑制する。
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)
