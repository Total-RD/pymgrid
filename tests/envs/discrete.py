import numpy as np

from copy import deepcopy

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.envs import DiscreteMicrogridEnv
from pymgrid.envs.base import BaseMicrogridEnv
from pymgrid.algos.priority_list import PriorityListAlgo, PriorityListElement
from pymgrid.modules import BatteryModule, GensetModule


class TestDiscreteEnv(TestCase):

    def _check_env(self, env, source_microgrid):
        self.assertIsInstance(env, PriorityListAlgo)
        self.assertIsInstance(env, BaseMicrogridEnv)

        self.assertEqual(env.modules, source_microgrid.modules)
        self.assertIsNot(env.modules.module_tuples(), source_microgrid.modules.module_tuples())

        n_obs = sum([x.observation_spaces['normalized'].shape[0] for x in source_microgrid.module_list])

        self.assertEqual(env.observation_space.shape, (n_obs,))

    def test_init_from_microgrid(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid)

        self._check_env(env, microgrid)

    def test_init_from_modules(self):
        microgrid = get_modular_microgrid()
        env = DiscreteMicrogridEnv(microgrid.modules.module_tuples(), add_unbalanced_module=False)

        self._check_env(env, microgrid)

    def test_populate_action_battery_grid(self, battery_first=True):
        battery = BatteryModule(min_capacity=0,
                                max_capacity=60,
                                max_charge=30,
                                max_discharge=60,
                                efficiency=1.0,
                                init_soc=1)

        microgrid = get_modular_microgrid(retain_only=('load', 'grid'), additional_modules=[battery])
        env = DiscreteMicrogridEnv.from_microgrid(microgrid)

        battery_element = PriorityListElement(module=('battery', 0), module_actions=1, action=0)
        grid_element = PriorityListElement(module=('grid', 0), module_actions=1, action=0)

        if battery_first:
            priority_list = (battery_element, grid_element)
            battery_val, grid_val = 60.0, 0.0
        else:
            priority_list = (grid_element, battery_element)
            grid_val, battery_val = 60.0, 0.0

        action = env._populate_action(priority_list)

        self.assertEqual(action['battery'], [battery_val])
        self.assertEqual(action['grid'], [grid_val])

    def test_populate_action_grid_battery(self):
        self.test_populate_action_battery_grid(battery_first=False)

    def test_populate_action_battery_genset_0_1(self, order=('battery', 'genset_0', 'genset_1')):
        battery = BatteryModule(min_capacity=0,
                                max_capacity=60,
                                max_charge=30,
                                max_discharge=60,
                                efficiency=1.0,
                                init_soc=1)

        genset = GensetModule(running_min_production=10, running_max_production=60, genset_cost=0.5)

        microgrid = get_modular_microgrid(retain_only=('load',), additional_modules=[battery, genset])
        env = DiscreteMicrogridEnv.from_microgrid(microgrid)

        elements = {
            'battery': PriorityListElement(module=('battery', 0), module_actions=1, action=0),
            'genset_0': PriorityListElement(module=('genset', 0), module_actions=2, action=0),
            'genset_1': PriorityListElement(module=('genset', 0), module_actions=2, action=1)
        }

        priority_list = [elements[element] for element in order]

        """
        battery_val should be 0 if genset_1 is ahead of battery AND genset_1 is ahead of genset_0"""
        if order.index('genset_1') < order.index('battery') and order.index('genset_1') < order.index('genset_0'):
            battery_val = 0.0
        else:
            battery_val = 60.0

        genset_val = np.array([
            int(order.index('genset_1') < order.index('genset_0')),
            60.0-battery_val
        ])

        expected_action = {'battery': [battery_val],
                           'genset': [genset_val]}

        action = env._populate_action(priority_list)

        self.assertEqual(action, expected_action)


    def test_populate_action_battery_genset_1_0(self):
        return self.test_populate_action_battery_genset_0_1(('battery', 'genset_1', 'genset_0'))

    def test_populate_action_genset_1_0_battery(self):
        return self.test_populate_action_battery_genset_0_1(('genset_1', 'battery', 'genset_0'))

    def test_populate_action_genset_0_1_battery(self):
        return self.test_populate_action_battery_genset_0_1(('genset_0', 'genset_1', 'battery'))

    def test_populate_action_genset_0_battery_genset_1(self):
        return self.test_populate_action_battery_genset_0_1(('genset_0', 'battery', 'genset_1'))

    def test_populate_action_genset_1_battery_genset_0(self):
        return self.test_populate_action_battery_genset_0_1(('genset_1', 'battery', 'genset_0'))


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
