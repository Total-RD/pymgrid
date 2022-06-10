from to_test_helpers.genset_module_testing_utils import default_params, get_genset, normalize_production
from to_test_helpers.test_case import TestCase
import numpy as np
from copy import deepcopy


class TestGensetStartUp2WindDown3OnAtStartUp(TestCase):
    def setUp(self) -> None:
        self.genset, self.default_params = get_genset(init_start_up=True, start_up_time=2, wind_down_time=3)

    def get_genset(self, **new_params):
        if len(new_params) == 0:
            return deepcopy(self.genset), self.default_params
        return get_genset(default_parameters=self.default_params, **new_params)

    def turn_on(self, genset, unnormalized_production=0.):
        # Take a step, ask genset to turn on.
        action = np.array([1.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        return obs, reward, done, info

    def turn_off(self, genset, unnormalized_production=50.):
        # Take a step, ask genset to turn on.
        action = np.array([0.0, normalize_production(unnormalized_production)])
        obs, reward, done, info = genset.step(action)
        return obs, reward, done, info

    def test_on_at_start_up(self):
        genset, _ = self.get_genset()
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(genset.current_obs, np.array([1, 1, 0, 3]))

    def test_turn_off_step_1(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.
        obs, reward, done, info = self.turn_off(genset, unnormalized_production)
        self.assertEqual(reward, -1.0*params['genset_cost']*unnormalized_production)
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([1, 0, 0, 2]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_turn_off_step_2(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(2):
            obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, -1.0*params['genset_cost']*unnormalized_production)
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([1, 0, 0, 1]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_turn_off_step_3(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(3):
            obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, -1.0*params['genset_cost']*unnormalized_production)
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([1, 0, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_turn_off_step_4_final(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(3):
            self.turn_off(genset, unnormalized_production)

        unnormalized_production = 0
        obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([0, 0, 2, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

    def test_turn_on_after_turn_off_step_1(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(3):
            self.turn_off(genset, unnormalized_production)

        # Step 4, should be off.
        unnormalized_production = 0
        obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([0, 0, 2, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

        unnormalized_production = 0
        obs, reward, done, info = self.turn_on(genset, unnormalized_production)
        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(genset.current_obs, np.array([0, 1, 1, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

    def test_turn_on_after_turn_off_step_2(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(3):
            self.turn_off(genset, unnormalized_production)

        # Step 4, should be off.
        unnormalized_production = 0
        obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([0, 0, 2, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

        # Turning back on
        unnormalized_production = 0
        for j in range(2):
            obs, reward, done, info = self.turn_on(genset, unnormalized_production)
        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(genset.current_obs, np.array([0, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

    def test_turn_on_after_turn_off_final(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(3):
            self.turn_off(genset, unnormalized_production)

        # Step 4, should be off.
        unnormalized_production = 0
        obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([0, 0, 2, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

        # Turning back on
        unnormalized_production = 0
        for j in range(2):
            self.turn_on(genset, unnormalized_production)

        unnormalized_production = 50.
        obs, reward, done, info = self.turn_on(genset, unnormalized_production)

        self.assertEqual(reward, -1.0*unnormalized_production*params['genset_cost'])
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(genset.current_obs, np.array([1, 1, 0, 3]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_turn_off_abortion(self):
        genset, params = self.get_genset()
        unnormalized_production = 50.

        for j in range(2):
            self.turn_off(genset, unnormalized_production)

        self.assertEqual(genset.current_obs, np.array([1, 0, 0, 1]))

        # Step 3: abort!
        obs, reward, done, info = self.turn_on(genset, unnormalized_production)

        self.assertEqual(reward, -1.0*unnormalized_production*params['genset_cost'])
        self.assertTrue(genset.is_running)
        self.assertEqual(genset.status_goal, 1)
        self.assertEqual(genset.current_obs, np.array([1, 1, 0, 3]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_turn_on_abortion(self):
        genset, params = self.get_genset(init_start_up=False)
        unnormalized_production = 0.

        self.turn_on(genset, unnormalized_production)
        self.assertEqual(genset.current_obs, np.array([0, 1, 1, 0]))

        # Step 3: abort!
        obs, reward, done, info = self.turn_off(genset, unnormalized_production)

        self.assertEqual(reward, 0)
        self.assertFalse(genset.is_running)
        self.assertEqual(genset.status_goal, 0)
        self.assertEqual(genset.current_obs, np.array([0, 0, 2, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)
