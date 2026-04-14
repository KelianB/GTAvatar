import time
import torch

class DotDict(dict):
    # Support dot notation
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def _get_time():
    torch.cuda.synchronize() # no significant difference
    # return time.time() # no significant difference
    return time.perf_counter()

class timeblock:
    """ Context manager for measuring execution time of code blocks.
    Usage:
    ```python
    with timeblock() as t:
        code_to_measure()
    print(t.get_ms()) # get time in milliseconds
    ```
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.time = None

    def __enter__(self):
        if self.enabled:
            self.start = _get_time()
        return self

    def __exit__(self, type, value, traceback):
        if self.enabled:
            self.time = _get_time() - self.start

    def get_ms(self):
        return None if self.time is None else self.time * 1000
    
    def get_s(self):
        return None if self.time is None else self.time