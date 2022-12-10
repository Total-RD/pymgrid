import numpy as np
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
            import warnings
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

    def contains(self, x):
        return x in self._unnormalized

    def sample(self, normalized=False):
        if normalized:
            return self._normalized.sample()
        return self._unnormalized.sample()

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
