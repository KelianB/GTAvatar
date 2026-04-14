import os
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    ''' Wrapper around tqdm that sets some values based on environment variables.'''

    mininterval = os.getenv("TQDM_MIN_INTERVAL", None)
    if mininterval is not None:
        mininterval = float(mininterval)

    miniters = os.getenv("TQDM_MIN_ITERS", None)
    if miniters is not None:
        miniters = int(miniters)

    return _tqdm(*args, mininterval=mininterval, miniters=miniters, **kwargs)