import numpy as np
from pymgrid.microgrid.modules.base import BaseMicrogridModule


class BaseTimeSeriesMicrogridModule(BaseMicrogridModule):
    def __init__(self, time_series,
                 raise_errors,
                 *args,
                 **kwargs):
        self._time_series = self._set_time_series(time_series)
        self._min_obs, self._max_obs, self._min_act, self._max_act = self.get_bounds()
        super().__init__(raise_errors,
                         *args,
                         **kwargs)

    def _set_time_series(self, time_series):
        _time_series = np.array(time_series)
        return _time_series

    def get_bounds(self):
        _min, _max = np.min(self._time_series), np.max(self._time_series)
        if self.is_sink and not self.is_source:
            _min, _max = -1*_max, -1*_min
            _max = 0.0
        else:
            _min = 0.0

        # if not (self.is_sink and self.is_source):
        #     return _min, 0.0, _max, _max

        return _min, _max, _min, _max

    @property
    def time_series(self):
        return self._time_series

    @property
    def min_obs(self):
        return self._min_obs

    @property
    def max_obs(self):
        return self._max_obs

    @property
    def min_act(self):
        return self._min_act

    @property
    def max_act(self):
        return self._max_act