import numpy as np
import warnings

from gym.spaces import Box, Space


class ModuleSpace(Space):
    """
    A space with both normalized and unnormalized versions.

    Both normalized and unormalized versions are of type :class:`gym:gym.spaces.Box`.

    This object also handles conversion of values from normalized to unnormalized and vice-versa.

    Parameters
    ----------
    unnormalized_low : float or array-like
        Lower bound of the space.

    unnormalized_high : float or array-like
        Upper bound of the space.

    shape : tuple or None, default None
        Shape of the space. If None, shape is inferred from `unnormalized_low` and `unnormalized_high`.

    dtype : np.dtype, default np.float64
        dtype of the action space.

    seed : int or None, default None
        Random seed.

        .. warning::
            This parameter will be ignored if earlier versions of `gym` are installed.

    """
    def __init__(self, unnormalized_low, unnormalized_high, shape=None, dtype=np.float64, seed=None):

        low = np.float64(unnormalized_low) if np.isscalar(unnormalized_low) else unnormalized_low.astype(np.float64)
        high = np.float64(unnormalized_high) if np.isscalar(unnormalized_high) else unnormalized_high.astype(np.float64)

        self._unnormalized = Box(low=low,
                                 high=high,
                                 shape=shape,
                                 dtype=dtype)

        self._normalized = Box(low=0, high=1, shape=self._unnormalized.shape, dtype=dtype)

        try:
            super().__init__(shape=self._unnormalized.shape, dtype=self._unnormalized.dtype, seed=seed)
        except TypeError:
            super().__init__(shape=self._unnormalized.shape, dtype=self._unnormalized.dtype)
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

        self._spread = self._unnormalized.high - self._unnormalized.low
        self._spread[self._spread == 0] = 1

    def contains(self, x):
        """
        Check if `x` is a valid member of the space.

        .. note::
            This method checks if `x` is a valid member of the space. Use :meth:`.ModuleSpace.normalized.contains` to
            check if a value is a member of the normalized space.

        Parameters
        ----------
        x : scalar or array-like
            Value to check membership.

        Returns
        -------
        containment : bool
            Whether `x` is a valid member of the space.

        """
        return x in self._unnormalized

    def sample(self, normalized=False):
        if normalized:
            return self._normalized.sample()
        return self._unnormalized.sample()

    def normalize(self, val):
        low, high = self._unnormalized.low, self._unnormalized.high

        self._shape_check(val, 'normalize')
        self._bounds_check(val, low, high)

        normalized = (val - low) / self._spread

        try:
            return normalized.item()
        except (AttributeError, ValueError):
            return normalized

    def denormalize(self, val):
        low, high = self._unnormalized.low, self._unnormalized.high

        self._shape_check(val, 'denormalize')
        self._bounds_check(val, 0, 1)

        denormalized = low + self._spread * val

        try:
            return denormalized.item()
        except (AttributeError, ValueError):
            return denormalized

    def _shape_check(self, val, func):
        low = self._unnormalized.low
        if hasattr(val, '__len__') and len(val) != len(low):
            raise TypeError(f'Unable to {func} value of length {len(val)}, expected {len(low)}')
        elif isinstance(val, (int, float)) and len(low) != 1:
            raise TypeError(f'Unable to {func} scalar value, expected array-like of shape {len(low)}')

    def _bounds_check(self, val, low, high):
        array_like_check = hasattr(val, '__len__') and \
                           not (all((low <= val) & (val <= high)) or np.allclose(val, low) or np.allclose(val, high))
        scalar_check = not hasattr(val, '__len__') and not low <= val <= high

        if array_like_check or scalar_check:
            warnings.warn(f'Value {val} resides out of expected bounds of value to be normalized: [{low}, {high}].')

    @property
    def normalized(self):
        return self._normalized

    @property
    def unnormalized(self):
        return self._unnormalized

    def __getitem__(self, item):
        if item == 'normalized':
            return self._normalized
        elif item == 'unnormalized':
            return self._unnormalized

        raise KeyError(item)

    def __repr__(self):
        return f'ModuleSpace{repr(self._unnormalized).replace("Box", "")}'

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.unnormalized == other.unnormalized
