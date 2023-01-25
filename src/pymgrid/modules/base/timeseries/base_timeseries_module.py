from abc import abstractmethod

import numpy as np
from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules.base import BaseMicrogridModule
from pymgrid.forecast.forecaster import get_forecaster, OracleForecaster, NoForecaster


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
        self._forecast_horizon = forecast_horizon * (forecaster is not None)
        self._forecaster = get_forecaster(forecaster,
                                          self._get_observation_spaces(),
                                          forecast_shape=(self.forecast_horizon, len(self.state_components)),
                                          time_series=self.time_series,
                                          increase_uncertainty=forecaster_increase_uncertainty)

        self._state_dict_keys = self._set_state_dict_keys()

        super().__init__(raise_errors, provided_energy_name=provided_energy_name,
                         absorbed_energy_name=absorbed_energy_name)

        self._current_forecast = self.forecast()

    def _set_time_series(self, time_series):
        _time_series = np.array(time_series)
        try:
            shape = (-1, _time_series.shape[1])
        except IndexError:
            shape = (-1, 1)
        _time_series = _time_series.reshape(shape)
        assert len(_time_series) == len(time_series)
        return self._sign_check(_time_series)

    def _sign_check(self, time_series):
        if self.is_source and self.is_sink:
            return time_series

        if not ((np.sign(time_series) <= 0).all() or (np.sign(time_series) >= 0).all()):
            raise ValueError('time_series cannot contain both positive and negative values unless it is both '
                             'a source and a sink.')

        if self.is_source:
            return np.abs(time_series)
        else:
            return -np.abs(time_series)

    def _get_bounds(self):
        _min, _max = np.min(self._time_series), np.max(self._time_series)
        if _min > 0:
            _min = 0
        elif _max < 0:
            _max = 0

        return _min, _max, _min, _max

    def _set_state_dict_keys(self):
        return {
            "current": [f"{component}_current" for component in self.state_components],
            "forecast": [
                f"{component}_forecast_{j}"
                for j in range(self._forecast_horizon) for component in self.state_components
            ]
        }

    def _update_step(self, reset=False):
        super()._update_step(reset=reset)
        self._current_forecast = self.forecast()

    def forecast(self):
        """
        Forecast the module's time series from the current state.

        Returns
        -------
        forecast : None or np.ndarray, shape (n, len(self.state_components))
            The forecasted time series.
        """
        val_c_n = self.time_series[1+self.current_step:1+self.current_step+self.forecast_horizon, :]
        try:
            val_c = self.time_series[self.current_step, :]
        except IndexError:
            forecast = self._forecaster.full_pad(self.time_series.shape, self._forecast_horizon)
        else:
            forecast = self._forecaster(val_c=val_c,
                                        val_c_n=val_c_n,
                                        n=self.forecast_horizon)

        return None if forecast is None else forecast

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
        try:
            return self.time_series[self.current_step, :]
        except IndexError:
            return self._forecaster.full_pad(self.time_series.shape, 1).reshape(-1)

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
        self._action_space = self._get_action_spaces()
        self._observation_space = self._get_observation_spaces()

    @property
    def min_obs(self):
        # TODO find a better solution
        return np.repeat(np.array(self._min_obs).reshape((-1, 1)), 1+self._forecast_horizon, axis=1).T.reshape(-1)

    @property
    def max_obs(self):
        # TODO find a better solution
        return np.repeat(np.array(self._max_obs).reshape((-1, 1)), 1+self._forecast_horizon, axis=1).T.reshape(-1)

    @property
    def min_act(self):
        return self._min_act

    @property
    def max_act(self):
        return self._max_act

    @property
    def forecaster(self):
        """
        View of the forecaster.

        Returns
        -------
        forecaster : :class:`.Forecaster`
            The module's forecaster.

        """
        return self._forecaster

    def set_forecaster(self,
                       forecaster,
                       forecast_horizon=DEFAULT_HORIZON,
                       forecaster_increase_uncertainty=False):
        """
        Set the forecaster for this module.

        Sets the forecaster with the same logic as upon initialization.

        forecaster : callable, float, "oracle", or None, default None.
            Function that gives a forecast n-steps ahead.

            * If ``callable``, must take as arguments ``(val_c: float, val_{c+n}: float, n: int)``, where

              * ``val_c`` is the current value in the time series: ``self.time_series[self.current_step]``

              * ``val_{c+n}`` is the value in the time series n steps in the future

              * n is the number of steps in the future at which we are forecasting.

              The output ``forecast = forecaster(val_c, val_{c+n}, n)`` must have the same sign
              as the inputs ``val_c`` and ``val_{c+n}``.

            * If ``float``, serves as a standard deviation for a mean-zero gaussian noise function
              that is added to the true value.

            * If ``"oracle"``, gives a perfect forecast.

            * If ``None``, no forecast.

        forecast_horizon : int
            Number of steps in the future to forecast. If forecaster is None, this parameter is ignored and the resultant
            horizon will be zero.

        forecaster_increase_uncertainty : bool, default False
            Whether to increase uncertainty for farther-out dates if using a GaussianNoiseForecaster. Ignored otherwise.

        """

        self.forecast_horizon = forecast_horizon * (forecaster is not None)

        self._forecaster = get_forecaster(forecaster,
                                          self._observation_space,
                                          (self.forecast_horizon, len(self.state_components)),
                                          self.time_series,
                                          increase_uncertainty=forecaster_increase_uncertainty)

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

        self._forecast_horizon = value
        self._state_dict_keys = self._set_state_dict_keys()
        self._observation_space = self._get_observation_spaces()

        if value > 0 and isinstance(self._forecaster, NoForecaster):
            from warnings import warn
            warn("Setting forecast_horizon requires a non-null forecaster. Implementing OracleForecaster.")
            self._forecaster = OracleForecaster(self._observation_space,
                                                forecast_shape=(value, len(self.state_components)),
                                                sink_only=self.is_sink and not self.is_source
                                                )

        self._forecaster.observation_space = self._observation_space

    @property
    def forecaster_increase_uncertainty(self):
        """
        View of :class:`pymgrid.forecast.GaussianNoiseForecaster``.increase_uncertainty`.

        Required for serialization as a mirror to the class parameter.
        Will only ever be True if ``self.forecaster`` is a ``GaussianNoiseForecaster``.

        Returns
        -------
        forecaster_increase_uncertainty : bool
            Associated attribute of ``self.forecaster``.

        """
        try:
            return self._forecaster.increase_uncertainty
        except AttributeError:
            return False

    @property
    def state_dict(self):
        state_dict = dict(zip(self._state_dict_keys['current'], self.current_obs))

        if self._current_forecast is not None:
            state_dict.update(zip(self._state_dict_keys['forecast'], self._current_forecast.reshape(-1)))

        return state_dict

    def serialize(self, dumper_stream):
        data = super().serialize(dumper_stream)
        data["cls_params"]["forecaster"] = self._forecast_param
        return data

    def serializable_state_attributes(self):
        return ["_current_step"]

    def __len__(self):
        return self._time_series.shape[0]
