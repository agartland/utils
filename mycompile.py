__all__ = ['mycompile']

def mycompile(filename):
    """Aid in the use of exec as replacement for execfile
    in Python 3

    Parameters
    ----------
    filename : str

    Returns
    -------
    code : code object
        For use with exec

    Usage
    -----
    
    >>> exec(mycompile(filename))

    Same as:

    >>> execfile(filename)"""

    return compile(open(filename).read(), filename, 'exec')
