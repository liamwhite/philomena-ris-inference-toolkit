import os
import signal


def _signal_handler(_signum, _stack_frame):
    os._exit(1)


def force_interrupt_shutdown():
    """
    Update console signal handlers so that the program can shut down
    when using multiple threads.
    """
    for s in [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM]:
        signal.signal(s, _signal_handler)


def set_base_path(base_path: str | None):
    """
    Set the current directory to the given path, if provided.
    """
    if base_path:
        os.chdir(base_path)
