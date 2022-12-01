def skip_init(cls, init):
    """
    Skip init once on cls, and then revert to original init.

    Parameters
    ----------
    cls : Type
        Class to skip init on.
    init : callable
        Original init.

    Returns
    -------
    skip_init : callable
        Callable that skips init once.

    """
    def reset_init(*args, **kwargs):
        cls.__init__ = init
    return reset_init
