import numpy as np
import pandas as pd


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
        self.microgrid, self.n_loads, self.n_pvs = self.set_microgrid()
        self.n_modules = 1 + self.n_loads + self.n_pvs

    def set_ts(self):
        ts = 10 * np.random.rand(100)
        return ts, ts

    def set_microgrid(self):
        load = LoadModule(time_series=self.load_ts, raise_errors=True)
        pv = RenewableModule(time_series=self.pv_ts, raise_errors=True)
        return Microgrid([load, pv]), 1, 1

    def test_populated_correctly(self):
        self.assertTrue(hasattr(self.microgrid.modules, 'load'))
        self.assertTrue(hasattr(self.microgrid.modules, 'renewable'))
        self.assertEqual(len(self.microgrid.modules), self.n_modules)  # load, pv, unbalanced

    def test_current_load_correct(self):
        try:
            current_load = self.microgrid.modules.load.item().current_load
        except ValueError:
            # More than one load module
            current_load = sum(load.current_load for load in self.microgrid.modules.load)
        self.assertEqual(current_load, self.load_ts[0])

    def test_current_pv_correct(self):
        try:
            current_renewable = self.microgrid.modules.renewable.item().current_renewable
        except ValueError:
            # More than one load module
            current_renewable = sum(renewable.current_renewable for renewable in self.microgrid.modules.renewable)
        self.assertEqual(current_renewable, self.pv_ts[0])

    def test_sample_action(self):
        sampled_action = self.microgrid.sample_action()
        self.assertEqual(len(sampled_action), 0)

    def test_sample_action_with_flex(self):
        sampled_action = self.microgrid.sample_action(sample_flex_modules=True)
        self.assertEqual(len(sampled_action), 2)
        self.assertIn('renewable', sampled_action)
        self.assertIn('balancing', sampled_action)
        self.assertEqual(len(sampled_action['renewable']), self.n_pvs)

    def test_state_dict(self):
        sd = self.microgrid.state_dict
        self.assertIn('load', sd)
        self.assertIn('renewable', sd)
        self.assertIn('balancing', sd)
        self.assertEqual(len(sd['load']), self.n_loads)
        self.assertEqual(len(sd['balancing']), 1)

    def test_state_series(self):
        ss = self.microgrid.state_series
        self.assertEqual({'load', 'renewable'}, set(ss.index.get_level_values(0)))
        self.assertEqual(ss['load'].index.get_level_values(0).nunique(), self.n_loads)
        self.assertEqual(ss['renewable'].index.get_level_values(0).nunique(), self.n_pvs)
        self.assertEqual(ss['load'].index.get_level_values(0).nunique(), self.n_loads)

    def test_to_nonmodular(self):
        if self.n_pvs > 1 or self.n_loads > 1:
            with self.assertRaises(ValueError) as e:
                self.microgrid.to_nonmodular()
                self.assertIn("Cannot convert modular microgrid with multiple modules of same type", e)

        else:
            nonmodular = self.microgrid.to_nonmodular()
            self.assertTrue(nonmodular.architecture['PV'])
            self.assertFalse(nonmodular.architecture['battery'])
            self.assertFalse(nonmodular.architecture['grid'])
            self.assertFalse(nonmodular.architecture['genset'])

    def check_step(self, microgrid, step_number=0):

        control = microgrid.get_empty_action()
        self.assertEqual(len(control), 0)

        obs, reward, done, info = microgrid.run(control)
        loss_load = self.load_ts[step_number]-self.pv_ts[step_number]
        loss_load_cost = self.microgrid.modules.balancing[0].loss_load_cost * max(loss_load, 0)

        self.assertEqual(loss_load_cost, -1*reward)

        self.assertEqual(len(microgrid.log), step_number + 1)
        self.assertTrue(all(module in microgrid.log for module in microgrid.modules.names()))

        load_met = min(self.load_ts[step_number], self.pv_ts[step_number])
        loss_load = max(self.load_ts[step_number] - load_met, 0)
        pv_curtailment = max(self.pv_ts[step_number]-load_met, 0)

        # Checking the log populated correctly.

        log_row = microgrid.log.iloc[step_number]
        log_entry = lambda module, entry: log_row.loc[pd.IndexSlice[module, :, entry]].sum()

        # Check that there are log entries for all modules of each name
        self.assertEqual(log_row['load'].index.get_level_values(0).nunique(), self.n_loads)

        self.assertEqual(log_entry('load', 'load_current'), -1 * self.load_ts[step_number])
        self.assertEqual(log_entry('load', 'load_met'), self.load_ts[step_number])

        if loss_load == 0:
            self.assertEqual(log_entry('load', 'load_met'), load_met)

        self.assertEqual(log_entry('renewable',  'renewable_current'), self.pv_ts[step_number])
        self.assertEqual(log_entry('renewable', 'renewable_used'), load_met)
        self.assertEqual(log_entry('renewable', 'curtailment'), pv_curtailment)

        self.assertEqual(log_entry('balancing', 'loss_load'), loss_load)

        self.assertEqual(log_entry('balance', 'reward'), -1 * loss_load_cost)
        self.assertEqual(log_entry('balance', 'overall_provided_to_microgrid'), self.load_ts[step_number])
        self.assertEqual(log_entry('balance', 'overall_absorbed_from_microgrid'), self.load_ts[step_number])
        self.assertEqual(log_entry('balance', 'fixed_provided_to_microgrid'), 0.0)
        self.assertEqual(log_entry('balance', 'fixed_absorbed_from_microgrid'), self.load_ts[step_number])
        self.assertEqual(log_entry('balance', 'controllable_absorbed_from_microgrid'), 0.0)
        self.assertEqual(log_entry('balance', 'controllable_provided_to_microgrid'), 0.0)

        return microgrid

    def test_run_one_step(self):
        microgrid = self.microgrid
        self.check_step(microgrid=microgrid, step_number=0)

    def test_run_n_steps(self):
        microgrid = self.microgrid
        for step in range(len(self.load_ts)):
            with self.subTest(step=step):
                microgrid = self.check_step(microgrid=microgrid, step_number=step)


class TestMicrogridLoadExcessPV(TestMicrogridLoadPV):
    #  Same as above but pv is greater than load.
    def set_ts(self):
        load_ts = 10*np.random.rand(100)
        pv_ts = load_ts + 5*np.random.rand(100)
        return load_ts, pv_ts


class TestMicrogridPVExcessLoad(TestMicrogridLoadPV):
    # Load greater than PV.
    def set_ts(self):
        pv_ts = 10 * np.random.rand(100)
        load_ts = pv_ts + 5 * np.random.rand(100)
        return load_ts, pv_ts


class TestMicrogridTwoLoads(TestMicrogridLoadPV):
    def set_microgrid(self):
        load_1_ts = self.load_ts*(1-np.random.rand(*self.load_ts.shape))
        load_2_ts = self.load_ts - load_1_ts

        assert all(load_1_ts > 0)
        assert all(load_2_ts > 0)

        load_1 = LoadModule(time_series=load_1_ts, raise_errors=True)
        load_2 = LoadModule(time_series=load_2_ts, raise_errors=True)
        pv = RenewableModule(time_series=self.pv_ts, raise_errors=True)
        return Microgrid([load_1, load_2, pv]), 2, 1


class TestMicrogridTwoPV(TestMicrogridLoadPV):
    def set_microgrid(self):
        pv_1_ts = self.pv_ts*(1-np.random.rand(*self.pv_ts.shape))
        pv_2_ts = self.pv_ts - pv_1_ts

        assert all(pv_1_ts > 0)
        assert all(pv_2_ts > 0)

        load = LoadModule(time_series=self.load_ts, raise_errors=True)
        pv_1 = RenewableModule(time_series=pv_1_ts, raise_errors=True)
        pv_2 = RenewableModule(time_series=pv_2_ts)
        return Microgrid([load, pv_1, pv_2]), 1, 2


class TestMicrogridTwoEach(TestMicrogridLoadPV):
    def set_microgrid(self):
        load_1_ts = self.load_ts*(1-np.random.rand(*self.load_ts.shape))
        load_2_ts = self.load_ts - load_1_ts

        pv_1_ts = self.pv_ts*(1-np.random.rand(*self.pv_ts.shape))
        pv_2_ts = self.pv_ts - pv_1_ts

        assert all(load_1_ts > 0)
        assert all(load_2_ts > 0)
        assert all(pv_1_ts > 0)
        assert all(pv_2_ts > 0)

        load_1 = LoadModule(time_series=load_1_ts, raise_errors=True)
        load_2 = LoadModule(time_series=load_2_ts, raise_errors=True)
        pv_1 = RenewableModule(time_series=pv_1_ts, raise_errors=True)
        pv_2 = RenewableModule(time_series=pv_2_ts)

        return Microgrid([load_1, load_2, pv_1, pv_2]), 2, 2


class TestMicrogridManyEach(TestMicrogridLoadPV):
    def set_microgrid(self):
        n_loads = np.random.randint(3, 10)
        n_pvs = np.random.randint(3, 10)

        load_ts = [self.load_ts * (1 - np.random.rand(*self.load_ts.shape))]
        pv_ts = [self.pv_ts * (1 - np.random.rand(*self.pv_ts.shape))]

        for ts_list, ts_sum, n_modules in zip(
                [load_ts, pv_ts],
                [self.load_ts, self.pv_ts],
                [n_loads, n_pvs]
        ):
            remaining = ts_sum-ts_list[0]

            for j in range(1, n_modules-1):
                ts_list.append(remaining*(1-np.random.rand(*ts_sum.shape)))
                assert all(ts_list[-1] > 0)
                remaining -= ts_list[-1]

            assert all(remaining > 0)
            ts_list.append(remaining)

        load_modules = [LoadModule(time_series=ts) for ts in load_ts]
        pv_modules = [RenewableModule(time_series=ts) for ts in pv_ts]

        return Microgrid([*load_modules, *pv_modules]), n_loads, n_pvs


class TestMicrogridManyEachExcessPV(TestMicrogridManyEach):
    def set_ts(self):
        load_ts = 10*np.random.rand(100)
        pv_ts = load_ts + 5*np.random.rand(100)
        return load_ts, pv_ts


class TestMicrogridManyEachExcessLoad(TestMicrogridManyEach):
    def set_ts(self):
        pv_ts = 10*np.random.rand(100)
        load_ts = pv_ts + 5*np.random.rand(100)
        return load_ts, pv_ts


class TestMicrogridRewardShaping(TestMicrogridLoadPV):
    def set_microgrid(self):
        original_microgrid, n_loads, n_pvs = super().set_microgrid()
        new_microgrid = Microgrid(original_microgrid.modules.to_tuples(),
                                  add_unbalanced_module=False,
                                  reward_shaping_func=self.reward_shaping_func)

        return new_microgrid, n_loads, n_pvs

    @staticmethod
    def reward_shaping_func(energy_info, cost_info):
        total = 0
        for module_name, info_list in energy_info.items():
            for j, (energy_type, energy_amount) in enumerate(info_list.items()):
                if energy_type == 'absorbed_energy':
                    marginal_cost = cost_info[module_name][j]['absorption_marginal_cost']
                elif energy_type == 'provided_energy':
                    marginal_cost = cost_info[module_name][j]['production_marginal_cost']
                else:
                    # Some other key
                    continue
                total += energy_amount * marginal_cost

        return total
