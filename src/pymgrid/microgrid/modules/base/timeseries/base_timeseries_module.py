from abc import ABC

import numpy as np
import logging
from pymgrid.microgrid.modules.base import BaseMicrogridModule
from pymgrid.microgrid.modules.base.timeseries.forecaster import get_forecaster

logger = logging.getLogger(__name__)


class BaseTimeSeriesMicrogridModule(BaseMicrogridModule, ABC):
    def __init__(self,
                 time_series,
                 raise_errors,
                 forecaster="oracle",
                 provided_energy_name='provided_energy',
                 absorbed_energy_name='absorbed_energy',
                 normalize_pos=...):
        self._time_series = self._set_time_series(time_series)
        self._min_obs, self._max_obs, self._min_act, self._max_act = self.get_bounds()
        self.forecaster = get_forecaster(forecaster, self.time_series)
        super().__init__(raise_errors,
                         provided_energy_name=provided_energy_name,
                         absorbed_energy_name=absorbed_energy_name,
                         normalize_pos=normalize_pos)

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
