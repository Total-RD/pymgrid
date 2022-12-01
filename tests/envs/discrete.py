from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import DiscreteMicrogridEnv


class TestDiscreteEnv(TestCase):
    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.module_tuples(), microgrid.modules.module_tuples())

        n_obs = sum([x.observation_spaces['normalized'].shape[0] for x in microgrid.module_list])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_init_from_modules(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid.modules.module_tuples(), add_unbalanced_module=False)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.module_tuples(), microgrid.modules.module_tuples())

        n_obs = sum([x.observation_spaces['normalized'].shape[0] for x in microgrid.module_list])

        self.assertEqual(env.observation_space.shape, (n_obs,))

