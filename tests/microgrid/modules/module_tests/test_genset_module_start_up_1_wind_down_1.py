from tests.helpers.genset_module_testing_utils import get_genset, normalize_production
from tests.helpers.test_case import TestCase
import numpy as np
from copy import deepcopy


class TestGensetStartUp1WindDown0OffAtStartUp(TestCase):
    def setUp(self) -> None:
        self.genset, self.default_params = get_genset(init_start_up=False, start_up_time=1)
        self.warm_up(self.genset)

    def get_genset(self, new=False, **new_params):
        if len(new_params) == 0 and not new:
            return deepcopy(self.genset), self.default_params

        genset, params = get_genset(default_parameters=self.default_params, **new_params)
        if not new:
            self.warm_up(genset)
        return genset, params


    def warm_up(self, genset):
        # Take a step, ask genset to turn on. Warm-up takes one step so genset is still off at this point.
        unnormalized_production = 0
        action = np.array([1.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        return obs, reward, done, info

    def test_off_at_start_up(self):
        genset, _ = self.get_genset(new=True)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([0, 0, 1, 0]))

    def test_warm_up(self):
        # Take a step, ask genset to turn on. Warm-up takes one step so genset is still off at this point.
        genset,_ = self.get_genset(new=True)
        obs, reward, done, info = self.warm_up(genset)
        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(obs, np.array([0, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

    def test_step_start_up_1_exception(self):
        # Assert that exception is thrown when production is requested while genset is off
        genset, _ = self.get_genset(new=True)

        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)

        unnormalized_production = 50
        action = np.array([1.0, normalize_production(unnormalized_production)])

        with self.assertRaises(ValueError) as e:
            genset.step(action)
        err_msg = e.exception.args[0]
        self.assertTrue('This may be because this genset module is not currently running.' in err_msg)

    def test_step_start_up_1_no_exception(self):
        # Genset is on now. Should be able to request production.

        genset, params = self.get_genset()

        unnormalized_production = 50
        action = np.array([1.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        self.assertEqual(reward, -1.0*params['genset_cost']*unnormalized_production)
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(obs, np.array([1, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_start_up_1_request_below_min_exception_raise(self):
        genset, params = self.get_genset()

        # Genset is on now. Requesting below min production.
        unnormalized_production = params['min_production']*np.random.rand()
        action = np.array([1.0, normalize_production(unnormalized_production)])
        with self.assertRaises(ValueError):
            genset.step(action)

    def test_start_up_1_request_below_min_no_exception(self):
        # Genset is on, requesting production less than the min should return min production.
        genset, params = self.get_genset(raise_errors=False)

        # Genset is on now. Requesting below min production.
        unnormalized_production = params['min_production']*np.random.rand()
        action = np.array([1.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        self.assertEqual(reward, -1.0*params['genset_cost']*params['min_production'])
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(obs, np.array([1, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], params['min_production'])

    def test_start_up_1_then_shut_down_exception_raise(self):
        # Genset is on, requesting production less than the min should return min production.
        genset, params = self.get_genset()

        # Genset is on now. Requesting below min production.
        unnormalized_production = 50.
        action = np.array([0.1, normalize_production(unnormalized_production)])
        with self.assertRaises(ValueError):
            genset.step(action)
        # self.assertEqual(reward, -1.0*params['genset_cost']*params['min_production'])
        # self.assert(genset.is_running)
        # self.assertEqual(genset.status_goal, 1)
        # self.assertEqual(obs, np.array([1, 1, 0, 0]))
        # self.assertFalse(done)
        # self.assertEqual(info['provided_energy'], params['min_production'])

    def test_start_up_1_then_shut_down_no_exception(self):
        # Genset is on, requesting production less than the min should return min production.
        genset, params = self.get_genset(raise_errors=False)

        # Genset is on now. Requesting below min production.
        unnormalized_production = 50.
        action = np.array([0.1, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        self.assertEqual(reward, 0.0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(obs, np.array([0, 0, params['start_up_time'], 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0.0)


class TestGensetStartUp1WindDown0OnAtStartUp(TestCase):
    def setUp(self) -> None:
        self.genset, self.default_params = get_genset(init_start_up=True, start_up_time=1)
        self.warm_up(self.genset)

    def get_genset(self, new=False, **new_params):
        if len(new_params) == 0 and not new:
            return deepcopy(self.genset), self.default_params

        genset, params = get_genset(default_parameters=self.default_params, **new_params)
        if not new:
            self.warm_up(genset)
        return genset, params

    def warm_up(self, genset):
        # Take a step, ask genset to turn on. Genset begins on so should be on at this point.
        unnormalized_production = self.default_params['min_production']
        action = np.array([1.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        return obs, reward, done, info

    def test_on_at_start_up(self):
        genset, _ = self.get_genset(new=True)
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(genset.current_obs, np.array([1, 1, 0, 0]))

    def test_warm_up(self):
        # Take a step, ask genset to turn on.  Genset begins on so should be on at this point.
        genset, params = self.get_genset(new=True)
        obs, reward, done, info = self.warm_up(genset)
        self.assertEqual(reward, -1.0*params['genset_cost']*params['min_production'])
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(obs, np.array([1, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], params['min_production'])

    def test_shut_down(self):
        genset, _ = self.get_genset()
        unnormalized_production = 0
        action = np.array([0.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(obs, np.array([0, 0, 1, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)