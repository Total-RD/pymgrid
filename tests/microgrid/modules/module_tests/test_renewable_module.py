import numpy as np
from gym.spaces import Box

from pymgrid.microgrid.modules.renewable_module import RenewableModule
from tests.helpers.test_case import TestCase


class TestRenewableModuleNoForecasting(TestCase):
    def setUp(self) -> None:
        self.time_series = 2-np.cos(np.pi*np.arange(100)/2)

    def test_init(self):
        renewable_module = RenewableModule(self.time_series)
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])
        self.assertIsNone(renewable_module.forecast())
        self.assertEqual(renewable_module.state, self.time_series[0])
        self.assertEqual(len(renewable_module.state_dict), 1)

    def test_action_space(self):
        renewable_module = RenewableModule(self.time_series)
        normalized_action_space = renewable_module.action_spaces["normalized"]
        unnormalized_action_space = renewable_module.action_spaces["unnormalized"]

        self.assertEqual(normalized_action_space, Box(low=0, high=1, shape=(1,)))
        self.assertEqual(unnormalized_action_space, Box(low=0, high=self.time_series.max(), shape=(1,)))

    def test_observation_space(self):
        renewable_module = RenewableModule(self.time_series)
        normalized_obs_space = renewable_module.observation_spaces["normalized"]
        unnormalized_obs_space = renewable_module.observation_spaces["unnormalized"]

        self.assertEqual(normalized_obs_space, Box(low=0, high=1, shape=(1,)))
        self.assertEqual(unnormalized_obs_space, Box(low=0, high=self.time_series.max(), shape=(1,)))

    def test_step(self):
        renewable_module = RenewableModule(self.time_series)
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])

        unnormalized_action = 1
        action = renewable_module.to_normalized(unnormalized_action, act=True)
        obs, reward, done, info = renewable_module.step(action)
        obs = renewable_module.from_normalized(obs, obs=True)
        self.assertEqual(obs, self.time_series[1])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["provided_energy"], unnormalized_action)
        self.assertEqual(info["curtailment"], 0)


class TestRenewableModuleForecasting(TestCase):
    def setUp(self) -> None:
        self.time_series = 2-np.cos(np.pi*np.arange(100)/2)
        self.forecast_horizon = 24

    def new_renewable_module(self):
        return RenewableModule(self.time_series, forecaster="oracle", forecast_horizon=self.forecast_horizon)

    def test_init(self):
        renewable_module = self.new_renewable_module()
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])
        self.assertIsNotNone(renewable_module.forecast())
        self.assertEqual(renewable_module.forecast(), self.time_series[1:1+self.forecast_horizon].reshape((-1, 1)))
        self.assertEqual(renewable_module.state, self.time_series[:1+self.forecast_horizon])
        self.assertEqual(len(renewable_module.state_dict), 1+self.forecast_horizon)

    def test_action_space(self):
        renewable_module = self.new_renewable_module()
        normalized_action_space = renewable_module.action_spaces["normalized"]
        unnormalized_action_space = renewable_module.action_spaces["unnormalized"]

        self.assertEqual(normalized_action_space, Box(low=0, high=1, shape=(1,)))
        self.assertEqual(unnormalized_action_space, Box(low=0, high=self.time_series.max(), shape=(1,)))

    def test_observation_space(self):
        renewable_module = self.new_renewable_module()
        normalized_obs_space = renewable_module.observation_spaces["normalized"]
        unnormalized_obs_space = renewable_module.observation_spaces["unnormalized"]

        self.assertEqual(normalized_obs_space, Box(low=0, high=1, shape=(1+renewable_module.forecast_horizon,)))
        self.assertEqual(unnormalized_obs_space, Box(low=0, high=self.time_series.max(), shape=(1+renewable_module.forecast_horizon,)))

    def test_step(self):
        renewable_module = self.new_renewable_module()
        self.assertEqual(renewable_module.current_renewable, self.time_series[0])

        unnormalized_action = 1
        action = renewable_module.to_normalized(unnormalized_action, act=True)
        obs, reward, done, info = renewable_module.step(action)
        obs = renewable_module.from_normalized(obs, obs=True)
        self.assertEqual(obs, self.time_series[1:self.forecast_horizon+2])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(info["provided_energy"], unnormalized_action)
        self.assertEqual(info["curtailment"], 0)
