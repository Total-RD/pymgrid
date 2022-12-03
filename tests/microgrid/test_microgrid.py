import numpy as np

from pymgrid import Microgrid

from tests.helpers.modular_microgrid import get_modular_microgrid
from tests.helpers.test_case import TestCase


class TestMicrogrid(TestCase):
    def test_from_scenario(self):
        for j in range(25):
            with self.subTest(microgrid_number=j):
                microgrid = Microgrid.from_scenario(j)
                self.assertTrue(hasattr(microgrid, "load"))
                self.assertTrue(hasattr(microgrid, "pv"))
                self.assertTrue(hasattr(microgrid, "battery"))
                self.assertTrue(hasattr(microgrid, "grid") or hasattr(microgrid, "genset"))

    def test_empty_action_without_load(self):
        microgrid = get_modular_microgrid(remove_modules=('load', ))
        action = microgrid.get_empty_action()

        self.assertIn('battery', action)
        self.assertIn('genset', action)
        self.assertIn('grid', action)

        self.assertTrue(all(v == [None] for v in action.values()))

    def test_empty_action_with_load(self):
        microgrid = get_modular_microgrid()
        action = microgrid.get_empty_action()

        self.assertIn('battery', action)
        self.assertIn('genset', action)
        self.assertIn('grid', action)
        self.assertNotIn('load', action)

        self.assertTrue(all(v == [None] for v in action.values()))

    def test_sample_action(self):
        microgrid = get_modular_microgrid()
        action = microgrid.sample_action()

        for module_name, action_list in action.items():
            for module_num, _act in enumerate(action_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    action_arr = np.array(_act)
                    if not action_arr.shape:
                        action_arr = np.array([_act])
                    self.assertTrue(microgrid.modules[module_name][module_num].action_space.shape[0])
                    self.assertIn(action_arr, microgrid.modules[module_name][module_num].action_space.normalized)

    def test_sample_action_all_modules_populated(self):
        microgrid = get_modular_microgrid()
        action = microgrid.sample_action()

        for module_name, module_list in microgrid.fixed.iterdict():
            for module_num, module in enumerate(module_list):
                with self.subTest(module_name=module_name, module_num=module_num):
                    empty_action_space = module.action_space.shape == (0, )
                    try:
                        _ = action[module_name][module_num]
                        has_corresponding_action = True
                    except KeyError:
                        has_corresponding_action = False

                    self.assertTrue(empty_action_space != has_corresponding_action)  # XOR
