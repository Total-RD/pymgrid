import numpy as np
import yaml

from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule


class LoadModule(BaseTimeSeriesMicrogridModule):
    """
    A renewable energy module.

    The classic examples of renewables are photovoltaics (PV) and wind turbines.

    Parameters
    ----------
    time_series : array-like, shape (n_steps, )
        Time series of load demand.

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

    forecast_horizon : int.
        Number of steps in the future to forecast. If forecaster is None, ignored and 0 is returned.

    forecaster_increase_uncertainty : bool, default False
        Whether to increase uncertainty for farther-out dates if using a GaussianNoiseForecaster. Ignored otherwise..

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    """
    module_type = ('load', 'fixed')
    yaml_tag = u"!LoadModule"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    state_components = np.array(["load"], dtype=object)

    def __init__(self,
                 time_series,
                 forecaster=None,
                 forecast_horizon=DEFAULT_HORIZON,
                 forecaster_increase_uncertainty=False,
                 raise_errors=False):
        super().__init__(time_series,
                         raise_errors=raise_errors,
                         forecaster=forecaster,
                         forecast_horizon=forecast_horizon,
                         forecaster_increase_uncertainty=forecaster_increase_uncertainty,
                         provided_energy_name=None,
                         absorbed_energy_name='load_met')

        self.name = ('load', None)

    def _get_bounds(self):
        _min_obs, _max_obs, _, _ = super()._get_bounds()
        return _min_obs, _max_obs, np.array([]), np.array([])

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_sink, f'Class {self.__class__.__name__} is a sink.'

        info = {'absorbed_energy': self.current_load}

        return 0.0, self._done(), info

    def sample_action(self, strict_bound=False):
        return np.array([])

    @property
    def max_consumption(self):
        return self.current_load

    @property
    def current_load(self):
        """
        Current load.

        Returns
        -------
        load : float
            Current load demand.

        """
        return -1 * self._time_series[self._current_step].item()

    @property
    def is_sink(self):
        return True
