from abc import ABC, abstractmethod

import numpy as np
from pymgrid.microgrid.modules.base import BaseMicrogridModule
from pymgrid.microgrid.modules.base.timeseries.forecaster import get_forecaster


class BaseTimeSeriesMicrogridModule(BaseMicrogridModule):
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
        self._forecast_param = forecaster
        self.forecaster, self._forecast_horizon = get_forecaster(forecaster,
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
        val_c_n = self.time_series[1+self.current_step:1+self.current_step+self.forecast_horizon, :]
        factor = -1 if (self.is_sink and not self.is_source) else 1
        try:
            val_c = self.time_series[self.current_step, :]
        except IndexError:
            return np.zeros((self._forecast_horizon, len(self.state_components)))

        forecast = self.forecaster(val_c=val_c,
                               val_c_n=val_c_n,
                               n=self.forecast_horizon)
        return None if forecast is None else factor * forecast

    def _done(self):
        return self._current_step == len(self) - 1

    @property
    def current_obs(self):
        factor = -1 if (self.is_sink and not self.is_source) else 1
        try:
            return factor * self.time_series[self.current_step, :]
        except IndexError:
            return np.zeros(self.time_series.shape[1])

    @property
    def time_series(self):
        return self._time_series

    @time_series.setter
    def time_series(self, value):
        self._time_series = self._set_time_series(value)
        self.get_bounds()

    @property
    def min_obs(self):
        return np.repeat([self._min_obs], 1+self._forecast_horizon)

    @property
    def max_obs(self):
        return np.repeat([self._max_obs], 1+self._forecast_horizon)

    @property
    def min_act(self):
        return self._min_act

    @property
    def max_act(self):
        return self._max_act

    @property
    def forecast_horizon(self):
        return self._forecast_horizon

    @forecast_horizon.setter
    def forecast_horizon(self, value):
        if self.forecaster is not None:
            self._forecast_horizon = value
        else:
            from warnings import warn
            from pymgrid.microgrid.modules.base.timeseries.forecaster import OracleForecaster
            warn("Setting forecast_horizon requires a non-null forecaster. Implementing OracleForecaster.")
            self.forecaster = OracleForecaster()
            self._forecast_horizon = value

    @property
    def forecaster_increase_uncertainty(self):
        try:
            return self.forecaster.increase_uncertainty
        except AttributeError:
            return False

    @property
    @abstractmethod
    def state_components(self):
        pass

    @property
    def state_dict(self):
        forecast = self.forecast()
        state_dict = dict(zip(self.state_components + "_current", self.current_obs))
        for j in range(0, self.forecast_horizon):
            state_dict.update(dict(zip(self.state_components + f'_forecast_{j}', forecast[j, :])))
        return state_dict

    def serialize(self, dumper_stream):
        data = super().serialize(dumper_stream)
        data["cls_params"]["forecaster"] = self._forecast_param
        return data

    def serializable_state_attributes(self):
        return ["_current_step"]

    def __len__(self):
        return self._time_series.shape[0]
