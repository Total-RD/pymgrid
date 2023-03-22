import operator
import numpy as np
import warnings

from abc import abstractmethod
from gym.spaces import Box, Dict, Space, Tuple
from typing import Union


class _PymgridDict(Dict):
    def __init__(self, d, act_or_obs, normalized=False, seed=None):
        builtins = self._extract_builtins(d, act_or_obs=act_or_obs, normalized=normalized)
        try:
            super().__init__(builtins, seed=seed)
        except AssertionError:
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

            super().__init__(builtins.spaces)

    def _extract_builtins(self, d, act_or_obs='act', normalized=False):
        if act_or_obs == 'act':
            spaces = self._extract_action_spaces(d)
        elif act_or_obs == 'obs':
            spaces = self._extract_observation_spaces(d)
        else:
            raise NameError(act_or_obs)

        return self._transform_builtins(spaces, normalized=normalized)

    def _extract_action_spaces(self, d):
        controllable = {}
        for module_name, module_list in d.items():
            controllable_spaces = [v['action_space'] for v in module_list if 'controllable' in v['module_type']]
            if controllable_spaces:
                controllable[module_name] = controllable_spaces

        return controllable

    def _extract_observation_spaces(self, d):
        obs_spaces = {}
        for module_name, module_list in d.items():
            spaces = [v['observation_space'] for v in module_list]
            if spaces:
                obs_spaces[module_name] = spaces

        return obs_spaces

    def _transform_builtins(self, d, normalized=False):
        space_key = 'normalized' if normalized else 'unnormalized'

        transformed = {}

        if isinstance(d, dict):
            transformed = {}
            for k, v in d.items():
                if isinstance(v, Space):
                    transformed[k] = v[space_key]
                else:
                    transformed[k] = self._transform_builtins(v, normalized=normalized)

            transformed = Dict(transformed)

        elif isinstance(d, list):
            transformed = []
            for v in d:
                if isinstance(v, Space):
                    transformed.append(v[space_key])
                else:
                    transformed.append(self._transform_builtins(v, normalized=normalized))

            transformed = Tuple(transformed)

        return transformed

    def get_attr(self, attr):
        return {k: [getattr(v, attr) for v in tup] for k, tup in self.items()}

    @property
    def low(self):
        return self.get_attr('low')

    @property
    def high(self):
        return self.get_attr('high')

    @property
    def shape(self):
        return self.get_attr('shape')

    @shape.setter
    def shape(self, value):
        """
        Necessary for compatability with gym<0.21.
        """
        assert value is None

    def __getattr__(self, item):
        """
        Necessary for compatability with gym<0.21, where gym.spaces.Dict did not inherit from collections.Mapping.
        """
        if item == 'spaces':
            raise AttributeError('spaces')
        try:
            return getattr(self.spaces, item)
        except AttributeError:
            raise AttributeError(item)


class _PymgridSpace(Space):
    _unnormalized: Union[Box, _PymgridDict]
    _normalized: Union[Box, _PymgridDict]
    _spread: Union[np.ndarray, dict]

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

    def _shape_check(self, val, func):
        low = self._unnormalized.low
        if hasattr(val, '__len__') and len(val) != len(low):
            raise TypeError(f'Unable to {func} value of length {len(val)}, expected {len(low)}')
        elif isinstance(val, (int, float)) and len(low) != 1:
            raise TypeError(f'Unable to {func} scalar value, expected array-like of shape {len(low)}')

    @abstractmethod
    def normalize(self, val):
        pass

    @abstractmethod
    def denormalize(self, val):
        pass

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


class ModuleSpace(_PymgridSpace):
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

    def _bounds_check(self, val, low, high):
        array_like_check = hasattr(val, '__len__') and \
                           not (all((low <= val) & (val <= high)) or np.allclose(val, low) or np.allclose(val, high))
        scalar_check = not hasattr(val, '__len__') and not low <= val <= high

        if array_like_check or scalar_check:
            warnings.warn(f'Value {val} resides out of expected bounds of value to be normalized: [{low}, {high}].')


class MicrogridSpace(_PymgridSpace):
    def __init__(self, module_space_dict, act_or_obs, seed=None):

        self._unnormalized = _PymgridDict(module_space_dict, act_or_obs)
        self._normalized = _PymgridDict(module_space_dict, act_or_obs, normalized=True)

        try:
            super().__init__(shape=None, seed=seed)

        except TypeError:
            super().__init__(shape=None)
            import gym
            warnings.warn(f"gym.Space does not accept argument 'seed' in version {gym.__version__}; this argument will "
                          f"be ignored. Upgrade your gym version with 'pip install -U gym' to use this functionality.")

        self._spread = self._get_spread()

    def _get_spread(self):
        spread = {}

        for k, high_list in self._unnormalized.high.items():
            low_list = self._unnormalized.low[k]
            s = [h-l for h, l in zip(high_list, low_list)]
            for s_val in s:
                s_val[s_val == 0] = 1

            spread[k] = s

        return spread

    def normalize(self, val):
        low, high = self._unnormalized.low, self._unnormalized.high

        self._shape_check(val, 'normalize')
        val_minus_low = self.dict_op(val, low, operator.sub)
        normalized = self.dict_op(val_minus_low, self._spread, operator.truediv)

        # normalized = (val - low) / self._spread

        return normalized

    def denormalize(self, val):
        low, high = self._unnormalized.low, self._unnormalized.high

        self._shape_check(val, 'denormalize')

        val_times_spread = self.dict_op(val, self._spread, operator.mul)
        denormalized = self.dict_op(val_times_spread, low, operator.add)

        return denormalized

    @staticmethod
    def dict_op(first, second, op):
        out = {}
        for k, first_list in first.items():
            second_list = second[k]
            out[k] = [op(f, s) for f, s in zip(first_list, second_list)]
        return out
