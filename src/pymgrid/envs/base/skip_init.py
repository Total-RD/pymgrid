def skip_init(cls, init):
    """
    Skip init once on cls, and then revert to original init.
    :param cls: Class to skip init on.
    :param init: original init.
    :return: callable that skips init once.
    """
    def reset_init(*args, **kwargs):
        cls.__init__ = init
    return reset_init
