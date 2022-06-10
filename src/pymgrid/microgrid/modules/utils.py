import numpy as np
import pandas as pd
from collections import UserDict
from warnings import warn

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
                    assert all((_min <= value) & (value <= _max))
                except TypeError:
                    assert np.isnan(value).all()
        except AssertionError:
            warn(f'Value {value} resides out of expected bounds of value to be normalized: [{_min}, {_max}].')


        normalized = (value - self._min_val) / self._spread
        try:
            return normalized.item()
        except (AttributeError, ValueError):
            return normalized

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


class ModuleLogger(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_length = 0

    def flush(self):
        d = self.data.copy()
        self.clear()
        return d

    def log(self, **log_dict):
        for key, value in log_dict.items():
            try:
                self[key].append(value)
            except KeyError:
                self[key] = [np.nan]*self._log_length
                self[key].append(value)

        self._log_length += 1

    def to_dict(self):
        return self.data.copy()

    def to_frame(self):
        return pd.DataFrame(self.data)
    #
    # def __getitem__(self, item):
    #     return getattr(self,item)
    #
    # def __setitem__(self, key, value):
    #     setattr(self, key, value)
    #
    # def __repr__(self):
    #     return str(self.to_dict())

    def __len__(self):
        return self._log_length
#
# logger = ModuleLogger()
# logger.log(apples=3)
# logger.log(apples=4, bananas=5)
# print('done')

# class MyDict(UserDict):
#
#   def __setitem__(self, key, value):
#     super().__setitem__(key, value * 10)
#
#
# d = MyDict(a=1, b=2)  # Bad! MyDict.__setitem__ not called
# d.update(c=3)  # Bad! MyDict.__setitem__ not called
# d['d'] = 4  # Good!
# print(d)  # {'a': 1, 'b': 2, 'c': 3, 'd': 40}
