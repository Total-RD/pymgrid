import numpy as np
from gym.spaces import Box, Space


class ModuleSpace(Space):
    def __init__(self, unnormalized_low, unnormalized_high, shape=None, dtype=np.float64, seed=None):

        self._unnormalized = Box(low=unnormalized_low.astype(np.float64),
                                 high=unnormalized_high.astype(np.float64),
                                 shape=shape,
                                 dtype=dtype)

        self._normalized = Box(low=0, high=1, shape=self._unnormalized.shape, dtype=dtype)
        super().__init__(self._unnormalized.shape, self._unnormalized.dtype, seed)

    def contains(self, x):
        return x in self._unnormalized

    def sample(self, normalized=False):
        if normalized:
            return self._normalized.sample()
        return self._unnormalized.sample()

    @property
    def dtype(self):
        return self._unnormalized.dtype

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
