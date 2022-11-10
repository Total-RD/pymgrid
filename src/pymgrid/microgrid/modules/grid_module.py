import numpy as np
import yaml

from pymgrid.microgrid.modules.base import BaseTimeSeriesMicrogridModule


class GridModule(BaseTimeSeriesMicrogridModule):
    """
    Module representing a grid

    :param max_import: float. Maximum import at any time step.

    :param max_export: float. Maximum export at any time step.

    :param time_series: array-like, shape (n_features, n_steps), n_features = {3, 4}.
        If n_features=3, time series of (import_price, export_price, co2_per_kwH) in each column, respectively.
            Grid is assumed to have no outages.
        If n_features=4, time series of (import_price, export_price, co2_per_kwH, grid_status)
            in each column, respectively. time_series[:, -1] -- the grid status -- must be binary.

    :param forecaster: callable, float, "oracle", or None, default None. Function that gives a forecast n-steps ahead.
        If callable, must take as arguments (val_c: float, val_{c+n}: float, n: int), where:
            val_c is the current value in the time series: self.time_series[self.current_step],
            val_{c+n} is the value in the time series n steps in the future,
            n is the number of steps in the future at which we are forecasting.
            The output forecast = forecaster(val_c, val_{c+n}, n) must have the same sign
            as the inputs val_c and val_{c+n}.

        If float, serves as a standard deviation for a mean-zero gaussian noise function
            that is added to the true value.

        If "oracle", gives a perfect forecast.

        If None, no forecast.

    :param forecast_horizon: int. Number of steps in the future to forecast. If forecaster is None, ignored and 0 is returned.

    :param forecaster_increase_uncertainty: bool, default False. Whether to increase uncertainty for farther-out dates if using
        a GaussianNoiseForecaster. Ignored otherwise.

    :param cost_per_unit_co2: float, default 0.0. Marginal cost of grid co2 production.

    :param raise_errors: bool, default False.
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    """
    module_type = ('grid', 'fixed')
    yaml_tag = u"!GridModule"
    yaml_loader = yaml.SafeLoader
    yaml_dumper = yaml.SafeDumper

    def __init__(self,
                 max_import,
                 max_export,
                 time_series,
                 forecaster=None,
                 forecast_horizon=24,
                 forecaster_increase_uncertainty=False,
                 cost_per_unit_co2=0.0,
                 raise_errors=False):

        time_series = self._check_params(max_import, max_export, time_series)
        self.max_import, self.max_export = max_import, max_export
        self.cost_per_unit_co2 = cost_per_unit_co2
        self.name = ('grid', None)
        super().__init__(time_series,
                         raise_errors,
                         forecaster=forecaster,
                         forecast_horizon=forecast_horizon,
                         forecaster_increase_uncertainty=forecaster_increase_uncertainty,
                         provided_energy_name='grid_import',
                         absorbed_energy_name='grid_export')

    def _check_params(self, max_import, max_export, time_series):
        if max_import < 0:
            raise ValueError('parameter max_import must be non-negative.')
        if max_export < 0:
            raise ValueError('parameter max_export must be non-negative.')
        if time_series.shape[1] not in [3, 4]:
            raise ValueError('Time series must be two dimensional with three or four columns.'
                             'See docstring for details.')

        if time_series.shape[1] == 4:
            if not ((np.array(time_series)[:, -1] == 0) | (np.array(time_series)[:, -1] == 1)).all():
                raise ValueError("Last column (grid status) must contain binary values.")
        else:
            new_ts = np.ones((time_series.shape[0], 4))
            new_ts[:, :3] = time_series
            time_series = new_ts

        if (time_series < 0).any().any():
            raise ValueError('Time series must be non-negative.')

        return time_series

    def get_bounds(self):
        min_obs = self._time_series.min(axis=0)
        max_obs = self._time_series.max(axis=0)
        assert len(min_obs) in (3, 4)

        min_act, max_act = -1 * self.max_export, self.max_import

        return min_obs, max_obs, min_act, max_act

    def update(self, external_energy_change, as_source=False, as_sink=False):
        assert as_source + as_sink == 1, 'Must act as either source or sink but not both or neither.'

        reward = self.get_cost(external_energy_change, as_source, as_sink)
        info_key = 'provided_energy' if as_source else 'absorbed_energy'
        info = {info_key: external_energy_change,
                'co2_production': self.get_co2_production(external_energy_change, as_source, as_sink)}

        return reward, self._done(), info

    def get_cost(self, import_export, as_source, as_sink):
        if as_source:                                               # Import
            import_cost = self._time_series[self.current_step, 0]
            return -1 * import_cost*import_export + self.get_co2_cost(import_export, as_source, as_sink)
        elif as_sink:                                               # Export
            export_cost = self._time_series[self.current_step, 1]
            return export_cost * import_export + self.get_co2_cost(import_export, as_source, as_sink)
        else:
            raise RuntimeError

    def get_co2_cost(self, import_export, as_source, as_sink):
        return -1.0 * self.cost_per_unit_co2*self.get_co2_production(import_export, as_source, as_sink)

    def get_co2_production(self, import_export, as_source, as_sink):
        if as_source:                                               # Import
            co2_prod_per_kWh = self._time_series[self.current_step, 2]
            co2 = import_export*co2_prod_per_kWh
            return co2
        elif as_sink:                                               # Export
            return 0.0
        else:
            raise RuntimeError

    def as_flex(self):
        self.__class__.module_type = (self.__class__.module_type[0], 'flex')

    def as_fixed(self):
        self.__class__.module_type = (self.__class__.module_type[0], 'fixed')

    def import_price(self):
        return self.state[::4]

    def export_price(self):
        return self.state[1::4]

    def co2_per_kwh(self):
        return self.state[2::4]

    def grid_status(self):
        return self.state[3::4]

    @property
    def current_status(self):
        return self.grid_status()[0]

    @property
    def state_components(self):
        return np.array(['import_price', 'export_price', 'co2_per_kwh', 'grid_status'], dtype=object)

    @property
    def max_production(self):
        return self.max_import * self.current_status

    @property
    def max_consumption(self):
        return self.max_export * self.current_status

    @property
    def is_source(self):
        return True

    @property
    def is_sink(self):
        return True

    @property
    def weak_grid(self):
        return self._time_series[:, -1].min() < 1

    def __repr__(self):
        return f'GridModule(max_import={self.max_import}, max_export={self.max_export})'
