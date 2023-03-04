import numpy as np

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import DiscreteMicrogridEnv


class TestTrajectory(TestCase):

    def check_initial_final_steps(self,
                                  env,
                                  expected_env_initial,
                                  expected_env_final,
                                  expected_module_initial,
                                  expected_module_final):

        self.assertEqual(env.initial_step, expected_env_initial)
        self.assertEqual(env.final_step, expected_env_final)

        env.reset()

        self.assertEqual(env.initial_step, expected_env_initial)
        self.assertEqual(env.final_step, expected_env_final)

        self.assertEqual(env.modules.get_attrs('initial_step', unique=True).item(), expected_module_initial)
        self.assertEqual(env.modules.get_attrs('final_step', unique=True).item(), expected_module_final)

    def test_none_trajectory(self):
        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)
        env = DiscreteMicrogridEnv(modules, trajectory_func=None)
        self.check_initial_final_steps(env, 0, timeseries_length, 0, timeseries_length)

    def test_deterministic_trajectory(self):
        deterministic_initial, deterministic_final = 10, 20

        def trajectory_func(initial_step, final_step):
            return deterministic_initial, deterministic_final

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)
        env = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

        self.check_initial_final_steps(env, 0, timeseries_length, deterministic_initial, deterministic_final)

    def test_stochastic_trajectory(self):
        def trajectory_func(initial_step, final_step):
            initial = np.random.randint(low=initial_step+1, high=final_step-2)
            final = np.random.randint(low=initial, high=final_step)
            return initial, final

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)
        env = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

        self.assertEqual(env.initial_step, 0)
        self.assertEqual(env.final_step, timeseries_length)

        env.reset()

        self.assertEqual(env.initial_step, 0)
        self.assertEqual(env.final_step, timeseries_length)

        self.assertGreater(env.modules.get_attrs('initial_step', unique=True).item(), 0)
        self.assertLess(env.modules.get_attrs('initial_step', unique=True).item(), timeseries_length)

        self.assertGreater(env.modules.get_attrs('final_step', unique=True).item(), 0)
        self.assertLess(env.modules.get_attrs('final_step', unique=True).item(), timeseries_length)

        self.assertLess(env.modules.get_attrs('initial_step', unique=True).item(),
                        env.modules.get_attrs('final_step', unique=True).item())

    def test_bad_trajectory_out_of_range(self):
        def trajectory_func(initial_step, final_step):
            return 10, 110

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        with self.assertRaises(ValueError):
            _ = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

    def test_bad_trajectory_bad_signature(self):
        def trajectory_func(initial_step):
            return 10, 110

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        with self.assertRaises(TypeError):
            _ = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

    def test_bad_trajectory_initial_gt_final(self):
        def trajectory_func(initial_step, final_step):
            return 20, 10

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        with self.assertRaises(ValueError):
            _ = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

    def test_bad_trajectory_scalar_output(self):
        def trajectory_func(initial_step, final_step):
            return 20

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        with self.assertRaises(TypeError):
            _ = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

    def test_bad_trajectory_too_many_outputs(self):
        def trajectory_func(initial_step, final_step):
            return 10, 20, 30

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        with self.assertRaises(TypeError):
            _ = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

    def test_bad_trajectory_wrong_output_types(self):
        def trajectory_func(initial_step, final_step):
            return 'abc', 10.0

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        with self.assertRaises(TypeError):
            _ = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

    def test_correct_trajectory_length(self):

        def trajectory_func(initial_step, final_step):
            trajectory_func.n_resets += 1
            return 10, 11+trajectory_func.n_resets

        trajectory_func.n_resets = 0

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)
        env = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)

        for correct_trajectory_length in range(3, 7):
            with self.subTest(correct_trajectory_length=correct_trajectory_length):
                env.reset()
                n_steps = 0
                done = False

                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())
                    n_steps += 1

                self.assertEqual(n_steps, correct_trajectory_length)

    def test_trajectory_serialization(self):
        import yaml
        from pymgrid.microgrid.trajectory import DeterministicTrajectory

        trajectory_func = DeterministicTrajectory(10, 20)

        timeseries_length = 100
        modules = get_modular_microgrid(timeseries_length=timeseries_length, modules_only=True)

        env = DiscreteMicrogridEnv(modules, trajectory_func=trajectory_func)
        env.reset()
        loaded_env = yaml.safe_load(yaml.safe_dump(env))

        self.assertIsNotNone(env.trajectory_func)
        self.assertEqual(env, loaded_env)
