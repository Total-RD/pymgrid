from abc import ABC, abstractmethod

import numpy as np
from src.pymgrid.microgrid.modules.base import BaseMicrogridModule
from src.pymgrid.microgrid.modules.base.timeseries.forecaster import get_forecaster


class BaseTimeSeriesMicrogridModule(BaseMicrogridModule, ABC):
    def __init__(self,
                 time_series,
                 raise_errors,
                 forecaster=None,
                 forecast_horizon=24,
                 forecaster_increase_uncertainty=False,
                 provided_energy_name='provided_energy',
                 absorbed_energy_name='absorbed_energy',
                 normalize_pos=...):
        self._time_series = self._set_time_series(time_series)
        self._min_obs, self._max_obs, self._min_act, self._max_act = self.get_bounds()
        self.forecaster, self.forecast_horizon = get_forecaster(forecaster,
                                                                forecast_horizon,
                                                                self.time_series,
                                                                increase_uncertainty=forecaster_increase_uncertainty)
        super().__init__(raise_errors,
                         provided_energy_name=provided_energy_name,
                         absorbed_energy_name=absorbed_energy_name,
                         normalize_pos=normalize_pos)

    def _set_time_series(self, time_series):
        _time_series = np.array(time_series)
        try:
            shape = (-1, _time_series.shape[1])
        except IndexError:
            shape = (-1, 1)
        _time_series = _time_series.reshape(shape)
        assert len(_time_series) == len(time_series)
        return _time_series

    def get_bounds(self):
        _min, _max = np.min(self._time_series), np.max(self._time_series)
        if self.is_sink and not self.is_source:
            _min, _max = -1*_max, -1*_min
            _max = 0.0
        else:
            _min = 0.0

        return _min, _max, _min, _max

    def forecast(self):
        val_c_n = self.time_series[self.current_step:self.current_step+self.forecast_horizon, :]
        return self.forecaster(val_c=self.time_series[self.current_step, :],
                               val_c_n=val_c_n,
                               n=self.forecast_horizon)

    @property
    def current_obs(self):
        return self.time_series[self.current_step, :]

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

    @property
    @abstractmethod
    def state_components(self):
        pass

    @property
    def state_dict(self):
        forecast = self.forecast()
        state_dict = dict(zip(self.state_components + "_current", self.current_obs))
        for j in range(1, self.forecast_horizon):
            state_dict.update(dict(zip(self.state_components + f'_{j}', forecast[:, j])))
        return state_dict

