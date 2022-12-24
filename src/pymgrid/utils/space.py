import numpy as np
import warnings

from gym.spaces import Box, Space


class ModuleSpace(Space):
    def __init__(self, unnormalized_low, unnormalized_high, shape=None, dtype=np.float64, seed=None):

        self._unnormalized = Box(low=unnormalized_low.astype(np.float64),
                                 high=unnormalized_high.astype(np.float64),
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
