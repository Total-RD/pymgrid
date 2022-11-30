from _warnings import warn
import numpy as np


class Normalize:

    def __init__(self, min_val, max_val):
        self._max_val, self._min_val = max_val, min_val
        self._spread = self._max_val - self._min_val
        try:
            if self._spread == 0:
                self._spread = 1
        except ValueError:
            self._spread[self._spread == 0] = 1

    def to_normalized(self, value):
        _min, _max = self._min_val, self._max_val
        try:
            try:
                assert _min <= value <= _max, f'Value {value} is outside of range [{_min}, {_max}]'
            except (ValueError, AssertionError):
                try:
                    assert np.allclose(_min, value) or np.allclose(_max, value)
                except AssertionError:
                    try:
                        assert ((_min[~np.isnan(value)] <= value[~np.isnan(value)]) &
                                (value[~np.isnan(value)] <= _max[~np.isnan(value)])).all()
                    except TypeError:
                        assert np.isnan(value).all()
        except AssertionError:
            warn(f'Value {value} resides out of expected bounds of value to be normalized: [{_min}, {_max}].')

        normalized = (value - self._min_val) / self._spread
        try:
            return normalized.item()
        except (AttributeError, ValueError):
            return normalized

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return np.isclose(self._min_val, other._min_val).all() and np.isclose(self._max_val, other._max_val).all()

    def from_normalized(self, value):
        try:
            try:
                assert 0 <= value <= 1
            except ValueError:
                assert all((0 <= value) & (value <= 1))
        except AssertionError:
            warn(f'Value {value} resides out of expected bounds of normalized value: [0, 1].')

        unnormalized = self._min_val + self._spread * value
        try:
            return unnormalized.item()
        except (AttributeError, ValueError):
            return unnormalized

    @property
    def max_val(self):
        return self._max_val

    @property
    def min_val(self):
        return self._min_val

    def __repr__(self):
        return f'Normalize(min={self._min_val}, max={self._max_val})'


class IdentityNormalize(Normalize):
    def __init__(self):
        super().__init__(0, 1)

    def to_normalized(self, value):
        return value

    def from_normalized(self, value):
        return value

    @property
    def max_val(self):
        return 'The maximum value of the identity normalizer is not well-defined.'

    @property
    def min_val(self):
        return 'The minimum value of the identity normalizer is not well-defined.'

    def __repr__(self):
        return f'IdentityNormalize()'