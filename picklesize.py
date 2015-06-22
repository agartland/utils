import pandas as pd
import tempfile
import os
import numpy as np

__all__ = ['dirPickledSize', 'getPickledSize']

def dirPickledSize(obj,exclude=[]):
    """For each attribute of obj (excluding those specified and those that start with '__'),
    compute the size using getPickledSize(obj) and return as a pandas Series of KBs"""
    return pd.Series({o:getPickledSize(getattr(obj,o))/1024. for o in dir(obj) if not np.any([o[:2]=='__', o in exclude, getattr(obj,o) is None])})

def getPickledSize(obj):
    """Pickle obj to a temp file and then read the file size using the OS and return KB"""
    with tempfile.NamedTemporaryFile() as fh:
        pickle.dump(obj,fh)
        fh.flush()
        s = os.path.getsize(fh.name)
    return s/1024.