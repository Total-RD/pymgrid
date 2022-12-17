from copy import deepcopy

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import DiscreteMicrogridEnv


class TestDiscreteEnv(TestCase):

    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        n_obs = sum([x.observation_spaces['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_init_from_modules(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid.modules.to_tuples(), add_unbalanced_module=False)

        self.assertEqual(env.modules, microgrid.modules)
        self.assertIsNot(env.modules.to_tuples(), microgrid.modules.to_tuples())

        n_obs = sum([x.observation_spaces['normalized'].shape[0] for x in microgrid.modules.to_list()])

        self.assertEqual(env.observation_space.shape, (n_obs,))


class TestDiscreteEnvScenario(TestCase):
    microgrid_number = 0

    def setUp(self) -> None:
        self.env = DiscreteMicrogridEnv.from_scenario(microgrid_number=self.microgrid_number)

    def test_run_once(self):
        env = deepcopy(self.env)
        # sample environment then get log
        self.assertEqual(len(env.log), 0)
        for j in range(10):
            with self.subTest(step=j):
                action = env.sample_action(strict_bound=True)
                env.step(action)
                self.assertEqual(len(env.log), j+1)

    def test_reset_after_run(self):
        env = deepcopy(self.env)
        env.step(env.sample_action(strict_bound=True))
        env.reset()
        self.assertEqual(len(env.log), 0)

    def test_run_again_after_reset(self):
        env = deepcopy(self.env)
        env.step(env.sample_action(strict_bound=True))

        self.assertEqual(len(env.log), 1)

        env.reset()

        self.assertEqual(len(env.log), 0)

        for j in range(10):
            with self.subTest(step=j):
                action = env.sample_action(strict_bound=True)
                env.step(action)
                self.assertEqual(len(env.log), j+1)


class TestDiscreteEnvScenario1(TestDiscreteEnvScenario):
    microgrid_number = 1


class TestDiscreteEnvScenario2(TestDiscreteEnvScenario):
    microgrid_number = 2


class TestDiscreteEnvScenario3(TestDiscreteEnvScenario):
    microgrid_number = 3


class TestDiscreteEnvScenario4(TestDiscreteEnvScenario):
    microgrid_number = 4


class TestDiscreteEnvScenario5(TestDiscreteEnvScenario):
    microgrid_number = 5


class TestDiscreteEnvScenario6(TestDiscreteEnvScenario):
    microgrid_number = 6


class TestDiscreteEnvScenario47(TestDiscreteEnvScenario):
    microgrid_number = 7


class TestDiscreteEnvScenario8(TestDiscreteEnvScenario):
    microgrid_number = 8


class TestDiscreteEnvScenario9(TestDiscreteEnvScenario):
    microgrid_number = 9


class TestDiscreteEnvScenario10(TestDiscreteEnvScenario):
    microgrid_number = 10


class TestDiscreteEnvScenario11(TestDiscreteEnvScenario):
    microgrid_number = 11


class TestDiscreteEnvScenario12(TestDiscreteEnvScenario):
    microgrid_number = 12


class TestDiscreteEnvScenario13(TestDiscreteEnvScenario):
    microgrid_number = 13


class TestDiscreteEnvScenario14(TestDiscreteEnvScenario):
    microgrid_number = 14
    
    
class TestDiscreteEnvScenario15(TestDiscreteEnvScenario):
    microgrid_number = 15
    

class TestDiscreteEnvScenario16(TestDiscreteEnvScenario):
    microgrid_number = 16
    

class TestDiscreteEnvScenario17(TestDiscreteEnvScenario):
    microgrid_number = 17
    

class TestDiscreteEnvScenario18(TestDiscreteEnvScenario):
    microgrid_number = 18
    

class TestDiscreteEnvScenario19(TestDiscreteEnvScenario):
    microgrid_number = 19
    

class TestDiscreteEnvScenario20(TestDiscreteEnvScenario):
    microgrid_number = 20


class TestDiscreteEnvScenario21(TestDiscreteEnvScenario):
    microgrid_number = 21


class TestDiscreteEnvScenario22(TestDiscreteEnvScenario):
    microgrid_number = 22


class TestDiscreteEnvScenario23(TestDiscreteEnvScenario):
    microgrid_number = 23


class TestDiscreteEnvScenario24(TestDiscreteEnvScenario):
    microgrid_number = 24
