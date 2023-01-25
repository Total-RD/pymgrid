import numpy as np
from abc import abstractmethod
from gym.spaces import Box

from pymgrid.utils.space import ModuleSpace

from tests.helpers.test_case import TestCase


class TestTimeseriesModule(TestCase):
    __test__ = False

    def setUp(self) -> None:
        self.module_time_series = self._get_module_time_series()
        self.time_series = self._get_time_series()

    def _get_module_time_series(self):
        return self._get_time_series()

    def _get_time_series(self):
        return 2 - np.cos(np.pi * np.arange(100) / 2)

    @abstractmethod
    def get_module(self):
        return NotImplemented

    def test_action_space(self):
        module = self.get_module()
        normalized_action_space = module.action_space["normalized"]
        unnormalized_action_space = module.action_space["unnormalized"]

        self.assertEqual(normalized_action_space, Box(low=0, high=1, shape=(1, )))
        self.assertEqual(unnormalized_action_space, Box(low=0, high=self.time_series.max(), shape=(1, )))

    def test_observation_space(self):
        module = self.get_module()
        normalized_obs_space = module.observation_space["normalized"]
        unnormalized_obs_space = module.observation_space["unnormalized"]

        self.assertEqual(normalized_obs_space, Box(low=0, high=1, shape=(1+self.forecast_horizon,)))
        self.assertEqual(unnormalized_obs_space, Box(low=0, high=self.time_series.max(), shape=(1+self.forecast_horizon,)))


    def test_observations_in_observation_space(self):
        module = self.get_module()

        observation_space = ModuleSpace(
            unnormalized_low=0,
            unnormalized_high=self.time_series.max(),
            shape=(1 + module.forecast_horizon,)
        )

        self.assertEqual(module.observation_space, observation_space)

        done = False
        while not done:
            obs, reward, done, info = module.step(module.action_space.sample(), normalized=False)
            if np.isscalar(obs):
                obs = np.array([obs])
            self.assertIn(obs, observation_space['normalized'])
            self.assertIn(module.state, observation_space['unnormalized'])


class TestTimeseriesModuleNoForecasting(TestTimeseriesModule):
    forecast_horizon = 0

    def test_init(self):
        module = self.get_module()
        self.assertIsNone(module.forecast())
        self.assertEqual(module.state, self.time_series[0])
        self.assertEqual(len(module.state_dict), 1+self.forecast_horizon)


class TestTimeseriesModuleForecasting(TestTimeseriesModule):
    forecast_horizon = 24

    def test_init(self):
        renewable_module = self.get_module()
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])
        self.assertIsNotNone(renewable_module.forecast())
        self.assertEqual(renewable_module.forecast(), self.time_series[1:1+self.forecast_horizon].reshape((-1, 1)))
        self.assertEqual(renewable_module.state, self.time_series[:1+self.forecast_horizon])
        self.assertEqual(len(renewable_module.state_dict), 1+self.forecast_horizon)


class TestTimeSeriesModuleNoForecastingNegativeVals(TestTimeseriesModuleNoForecasting):
    def _get_module_time_series(self):
        return -1 * self._get_time_series()


class TestTimeSeriesModuleForecastingNegativeVals(TestTimeseriesModuleForecasting):
    def _get_module_time_series(self):
        return -1 * self._get_time_series()
