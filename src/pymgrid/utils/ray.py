import functools
from copy import copy


def ray_decorator(func):
    """
    ray raises an error when assigning values after ray.get to variables defined before ray.get.
    Easiest solution is to copy the values and try again.

    :meta private:

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'assignment destination is read-only' not in e.args[0]:
                raise
            else:
                return func(*(copy(a) for a in args), **{k: copy(v) for k, v in kwargs.items()})
    return wrapper
