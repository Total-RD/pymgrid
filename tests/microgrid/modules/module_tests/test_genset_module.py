import numpy as np

from tests.helpers.genset_module_testing_utils import default_params, get_genset, normalize_production
from tests.helpers.test_case import TestCase


class TestGensetModule(TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.default_params = default_params.copy()

    def get_genset(self, **new_params):
        return get_genset(**new_params)

    def test_init_start_up(self):
        genset, _ = self.get_genset()
        self.assertTrue(genset.current_status)
        genset, _ = self.get_genset(init_start_up=False)
        self.assertFalse(genset.current_status)

    def test_get_cost_linear(self):
        genset_cost = np.random.rand()
        genset, params = self.get_genset(genset_cost=genset_cost)

        production = params['running_min_production'] + (params['running_max_production']-params['running_min_production'])*np.random.rand()
        production_cost = production*genset_cost

        self.assertEqual(genset.get_cost(production), production_cost)

    def test_get_cost_callable(self):
        genset_cost = lambda x: x**2
        genset, params = self.get_genset(genset_cost=genset_cost)

        production = params['running_min_production'] + (params['running_max_production']-params['running_min_production'])*np.random.rand()
        production_cost = genset_cost(production)

        self.assertEqual(genset.get_cost(production), production_cost)

    def test_step_out_of_range_goal_status(self):
        genset, _ = self.get_genset()
        action = np.array([-0.5, 0.5])

        with self.assertRaises(AssertionError):
            genset.step(action)

    def test_step_out_of_normalized_range_production(self):
        genset, _ = self.get_genset()

        with self.assertRaises(AssertionError):
            action = np.array([-0.5, 2])
            genset.step(action)

    def test_step_incorrect_action_shape(self):
        genset, _ = self.get_genset()

        with self.assertRaises(TypeError):
            action = 0.5
            genset.step(action)

        with self.assertRaises(TypeError):
            action = np.ones(3)
            genset.step(action)

    def test_step_unnormalized_production(self):
        genset, _ = self.get_genset()

        action = np.array([1.0, 50])
        # try:
        obs, reward, done, info = genset.step(action, normalized=False)

        self.assertEqual(reward, -1.0 * default_params['genset_cost']*action[1])
        self.assertTrue(genset.current_status)
        self.assertEqual(genset.goal_status, 1)
        self.assertEqual(obs, np.array([1, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], action[1])

    def test_step_normalized_production(self):
        genset, params = self.get_genset()

        unnormalized_production = 50
        action = np.array([1.0, normalize_production(unnormalized_production)])
        # try:
        obs, reward, done, info = genset.step(action, normalized=True)

        self.assertEqual(reward, -1.0 * params['genset_cost']*unnormalized_production)
        self.assertTrue(genset.current_status)
        self.assertEqual(genset.goal_status, 1)
        self.assertEqual(obs, np.array([1, 1, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_step_immediate_status_change(self):
        genset, params = self.get_genset()

        unnormalized_production = 0
        action = np.array([0.0, normalize_production(unnormalized_production)])

        self.assertTrue(genset.current_status)

        obs, reward, done, info = genset.step(action, normalized=True)

        self.assertEqual(reward, 0)
        self.assertFalse(genset.current_status)
        self.assertEqual(genset.goal_status, 0)
        self.assertEqual(obs, np.array([0, 0, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], unnormalized_production)

    def test_step_genset_off_production_request_error_raise(self):
        genset, _ = self.get_genset()

        unnormalized_production = 50
        action = np.array([0.0, normalize_production(unnormalized_production)])

        # Genset starts on
        self.assertTrue(genset.current_status)
        self.assertEqual(genset.goal_status, 1)

        # Turn genset off (wind_down_time=0), and then ask for production. (no-no).
        with self.assertRaises(ValueError):
            genset.step(action, normalized=True)

    def test_step_genset_off_production_request_no_error_raise(self):
        genset, _ = self.get_genset(raise_errors=False)

        unnormalized_production = 50
        action = np.array([0.0, normalize_production(unnormalized_production)])

        obs, reward, done, info = genset.step(action)
        self.assertEqual(reward, 0)
        self.assertFalse(genset.current_status)
        self.assertEqual(genset.goal_status, 0)
        self.assertEqual(obs, np.array([0, 0, 0, 0]))
        self.assertFalse(done)
        self.assertEqual(info['provided_energy'], 0)

    def test_step_genset_production_request_out_of_range_no_error_raise(self):
        genset, params = self.get_genset(raise_errors=False)

        requested_possible_warn =  [(params['running_min_production']*np.random.rand(), params['running_min_production'], False),
                               (params['running_max_production'] * (1+np.random.rand()), params['running_max_production'], True)]


        # First requested value is below min_production, second is above max_production.
        # Second should raise a warning.

        for requested, possible, raises_warn in requested_possible_warn:
            with self.subTest(requested_production=requested, possible_production=possible):
                action = np.array([1.0, normalize_production(requested)])

                if raises_warn:
                    with self.assertWarns(Warning):
                        obs, reward, done, info = genset.step(action)
                else:
                    obs, reward, done, info = genset.step(action)

                self.assertEqual(reward, -1.0 * params['genset_cost']*possible)
                self.assertTrue(genset.current_status)
                self.assertEqual(genset.goal_status, 1)
                self.assertEqual(obs, np.array([1, 1, 0, 0]))
                self.assertFalse(done)
                self.assertEqual(info['provided_energy'], possible)
