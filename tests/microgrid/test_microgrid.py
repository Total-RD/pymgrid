import numpy as np

from pymgrid import Microgrid
from pymgrid.modules import LoadModule, RenewableModule

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


class TestMicrogridLoadPV(TestCase):
    def setUp(self):
        self.load_ts, self.pv_ts = self.set_ts()
        self.microgrid = self.set_microgrid()

    def set_ts(self):
        ts = 10 * np.random.rand(100)
        return ts, ts

    def set_microgrid(self):
        load = LoadModule(time_series=self.load_ts, raise_errors=True)
        pv = RenewableModule(time_series=self.pv_ts, raise_errors=True)
        return Microgrid([load, pv])

    def test_populated_correctly(self):
        self.assertTrue(hasattr(self.microgrid.modules, 'load'))
        self.assertTrue(hasattr(self.microgrid.modules, 'renewable'))
        self.assertEqual(len(self.microgrid.modules), 3)  # load, pv, unbalanced

    def test_current_load_correct(self):
        self.assertEqual(self.microgrid.modules.load[0].current_load, self.load_ts[0])

    def test_current_pv_correct(self):
        self.assertEqual(self.microgrid.modules.renewable[0].current_renewable, self.pv_ts[0])

    def test_run_one_step(self):
        pass

    def test_run_n_steps(self):
        for step in range(len(self.load_ts)):
            with self.subTest(step=step):
                pass


class TestMicrogridLoadExcessPV(TestMicrogridLoadPV):
    #  Same as above but pv is greater than load.
    def set_ts(self):
        load_ts = 10*np.random.rand(100)
        pv_ts = load_ts + 5*np.random.rand(100)
        return load_ts, pv_ts

    def test_run_one_step(self):
        pass

    def test_run_n_steps(self):
        for step in range(len(self.load_ts)):
            with self.subTest(step=step):
                pass


class TestMicrogridPVExcessLoad(TestMicrogridLoadPV):
    # Load greater than PV.
    def set_ts(self):
        pv_ts = 10 * np.random.rand(100)
        load_ts = pv_ts + 5 * np.random.rand(100)
        return load_ts, pv_ts

    def test_run_one_step(self):
        pass

    def test_run_n_steps(self):
        for step in range(len(self.load_ts)):
            with self.subTest(step=step):
                pass
