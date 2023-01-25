import numpy as np

from pymgrid.modules import LoadModule

from tests.microgrid.modules.module_tests.timeseries_modules import (
    TestTimeseriesModuleForecasting,
    TestTimeseriesModuleNoForecasting,
    TestTimeSeriesModuleForecastingNegativeVals,
    TestTimeSeriesModuleNoForecastingNegativeVals
)


class TestLoadModuleNoForecasting(TestTimeseriesModuleNoForecasting):
    __test__ = True
    negative_time_series = True
    action_space_dim = 0


    def get_module(self):
        return LoadModule(self.module_time_series)

    def test_init_current_load(self):
        load_module = self.get_module()
        self.assertEqual(load_module.current_load, -1 * self.time_series[0])

    def test_step(self):
        load_module = self.get_module()
        self.assertEqual(load_module.current_load, -1 * self.time_series[0])

        obs, reward, done, info = load_module.step(np.array([]))
        obs = load_module.from_normalized(obs, obs=True)
        self.assertEqual(obs, self.time_series[1])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["absorbed_energy"], -1 * self.time_series[0])


class TestLoadModuleForecasting(TestTimeseriesModuleForecasting):
    __test__ = True
    negative_time_series = True
    action_space_dim = 0

    def get_module(self):
        return LoadModule(self.module_time_series, forecaster="oracle", forecast_horizon=self.forecast_horizon)

    def test_step(self):
        load_module = self.get_module()
        self.assertEqual(load_module.current_load, -1 * self.time_series[0])

        action = load_module.to_normalized(np.array([]), act=True)
        obs, reward, done, info = load_module.step(action)
        obs = load_module.from_normalized(obs, obs=True)
        self.assertEqual(obs, self.time_series[1:self.forecast_horizon+2])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["absorbed_energy"], -1 * self.time_series[0])


class TestLoadModuleForecastingNegativeVals(TestTimeSeriesModuleForecastingNegativeVals,
                                            TestLoadModuleForecasting):
    pass


class TestLoadModuleNoForecastingNegativeVals(TestTimeSeriesModuleNoForecastingNegativeVals,
                                              TestLoadModuleNoForecasting):
    pass

