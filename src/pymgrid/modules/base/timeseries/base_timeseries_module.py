from abc import abstractmethod

import numpy as np
from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules.base import BaseMicrogridModule
from pymgrid.forecast.forecaster import get_forecaster


class BaseTimeSeriesMicrogridModule(BaseMicrogridModule):

    state_components = None
    """
    Labels of the components of each entry in the module's time series.

    Column labels of self.time_series.

    Returns
    -------
    state_components : np.ndarray[str], shape (self.time_series.shape[1], )
        The state components.
    """

    def __init__(self,
                 time_series,
                 raise_errors,
                 forecaster=None,
                 forecast_horizon=DEFAULT_HORIZON,
                 forecaster_increase_uncertainty=False,
                 provided_energy_name='provided_energy',
                 absorbed_energy_name='absorbed_energy',
                 normalize_pos=...):
        self._time_series = self._set_time_series(time_series)
        self._min_obs, self._max_obs, self._min_act, self._max_act = self._get_bounds()
        self._forecast_param = forecaster
        self.forecaster, self._forecast_horizon = get_forecaster(forecaster,
                                                                 forecast_horizon,
                                                                 self.time_series,
                                                                 increase_uncertainty=forecaster_increase_uncertainty)

        self._state_dict_keys = {"current":  [f"{component}_current" for component in self.state_components],
                                 "forecast": [
                                     f"{component}_forecast_{j}"
                                     for j in range(self._forecast_horizon) for component in self.state_components
                                 ]
                                 }

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

    def _get_bounds(self):
        _min, _max = np.min(self._time_series), np.max(self._time_series)
        if self.is_sink and not self.is_source:
            _min, _max = -1*_max, -1*_min
            _max = 0.0
        else:
            _min = 0.0

        return _min, _max, _min, _max

    def forecast(self):
        """
        Forecast the module's time series from the current state.

        Returns
        -------
        forecast : None or np.ndarray, shape (n, len(self.state_components))
            The forecasted time series.
        """
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
        """
        Current observation.

        Returns
        -------
        obs : np.ndarray, shape (len(self.state_components), )
            The observation.
        """
        factor = -1 if (self.is_sink and not self.is_source) else 1
        try:
            return factor * self.time_series[self.current_step, :]
        except IndexError:
            return np.zeros(self.time_series.shape[1])

    @property
    def time_series(self):
        """
        View of the module's time series.

        Returns
        -------
        time_series : np.ndarray, shape (len(self), len(self.state_components))
            The underlying time series.

        """
        return self._time_series

    @time_series.setter
    def time_series(self, value):
        self._time_series = self._set_time_series(value)
        self._min_obs, self._max_obs, self._min_act, self._max_act = self._get_bounds()
        self._obs_normalizer = self._get_normalizer(..., obs=True)
        self._act_normalizer = self._get_normalizer(..., act=True)

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
        """
        The number of steps until which the module forecasts.

        Returns
        -------
        forecast_horizon : int
            The forecast horizon.

        """
        return self._forecast_horizon

    @forecast_horizon.setter
    def forecast_horizon(self, value):
        if self.forecaster is not None:
            self._forecast_horizon = value
        else:
            from warnings import warn
            from pymgrid.forecast.forecaster import OracleForecaster
            warn("Setting forecast_horizon requires a non-null forecaster. Implementing OracleForecaster.")
            self.forecaster = OracleForecaster()
            self._forecast_horizon = value

    @property
    def forecaster_increase_uncertainty(self):
        """
        View of ``self.forecaster.increase_uncertainty``.

        Required for serialization as a mirror to the class parameter.
        Will only ever be True if ``self.forecaster`` is a ``GaussianNoiseForecaster``.

        Returns
        -------
        forecaster_increase_uncertainty : bool
            Associated attribute of ``self.forecaster``.

        """
        try:
            return self.forecaster.increase_uncertainty
        except AttributeError:
            return False

    @property
    def state_dict(self):
        forecast = self.forecast()

        state_dict = dict(zip(self._state_dict_keys['current'], self.current_obs))
        if forecast is not None:
            state_dict.update(zip(self._state_dict_keys['forecast'], forecast.reshape(-1)))

        return state_dict

    def serialize(self, dumper_stream):
        data = super().serialize(dumper_stream)
        data["cls_params"]["forecaster"] = self._forecast_param
        return data

    def serializable_state_attributes(self):
        return ["_current_step"]

    def __len__(self):
        return self._time_series.shape[0]
