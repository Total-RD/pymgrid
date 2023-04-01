from tests.helpers.test_case import TestCase

from pymgrid.modules import BatteryModule
from pymgrid.modules.battery.transition_models import BiasedTransitionModel, DecayTransitionModel

DEFAULT_PARAMS = {
            'min_capacity': 0,
            'max_capacity': 100,
            'max_charge': 50,
            'max_discharge': 50,
            'efficiency': 0.5,
            'battery_cost_cycle': 0.0,
            'battery_transition_model': None,
            'init_soc': 0.5
        }


def get_battery(**params):
    p = DEFAULT_PARAMS.copy()
    p.update(params)

    if 'init_charge' in params and 'init_soc' not in params:
        p.pop('init_soc')

    return BatteryModule(**p)


class TestBatteryModule(TestCase):
    def test_min_act(self):
        params = {
            'init_soc': 0,
            'efficiency': 0.5,
            'max_charge': 40,
            'max_discharge': 60
        }

        battery = get_battery(**params)
        expected_min_act = -1 * params['max_charge'] / params['efficiency']

        self.assertEqual(battery.soc, 0)
        self.assertEqual(battery.current_charge, 0)
        self.assertEqual(battery.min_act, expected_min_act)

        obs, reward, done, info = battery.step(expected_min_act, normalized=False)

        self.assertEqual(info['absorbed_energy'], -1 * expected_min_act)
        self.assertEqual(battery.current_charge, params['max_charge'])

    def test_max_act(self):
        params = {
            'init_soc': 1,
            'efficiency': 0.5,
            'max_charge': 40,
            'max_discharge': 60
        }

        battery = get_battery(**params)
        expected_max_act = params['max_discharge'] * params['efficiency']

        self.assertEqual(battery.soc, 1)
        self.assertEqual(battery.current_charge, DEFAULT_PARAMS['max_capacity'])
        self.assertEqual(battery.max_act, expected_max_act)

        obs, reward, done, info = battery.step(expected_max_act, normalized=False)

        self.assertEqual(info['provided_energy'], expected_max_act)
        self.assertEqual(battery.current_charge, 100 - params['max_discharge'])

    def test_max_consumption_max_charge(self):
        params = {
            'init_soc': 0,
            'efficiency': 0.5,
            'max_charge': 40,
            'max_discharge': 60
        }

        battery = get_battery(**params)

        self.assertEqual(battery.max_consumption, params['max_charge'] / params['efficiency'])

    def test_max_consumption_nonmax_charge(self):
        params = {
            'init_charge': 80,
            'efficiency': 0.5,
            'max_charge': 40,
            'max_discharge': 60
        }

        battery = get_battery(**params)

        self.assertEqual(
            battery.max_consumption,
            (DEFAULT_PARAMS['max_capacity']-params['init_charge']) / params['efficiency']
        )

    def test_max_production_max_discharge(self):
        params = {
            'init_soc': 1,
            'efficiency': 0.5,
            'max_charge': 40,
            'max_discharge': 60
        }

        battery = get_battery(**params)

        self.assertEqual(battery.max_production, params['max_discharge'] * params['efficiency'])

    def test_max_production_nonmax_discharge(self):
        params = {
            'init_charge': 20,
            'efficiency': 0.5,
            'max_charge': 40,
            'max_discharge': 60
        }

        battery = get_battery(**params)

        self.assertEqual(
            battery.max_production,
            (params['init_charge']-DEFAULT_PARAMS['min_capacity']) * params['efficiency']
        )


class TestBiasedBatteryModuleCharge(TestCase):
    def test_single_step(self):
        true_efficiency = 0.6
        init_soc = 1.0

        battery_transition_model = BiasedTransitionModel(true_efficiency=true_efficiency)
        battery = get_battery(battery_transition_model=battery_transition_model, init_soc=init_soc)

        self.assertEqual(battery.battery_transition_model.true_efficiency, true_efficiency)
        self.assertEqual(battery.battery_transition_model, battery_transition_model)

        _, _, _, info = battery.step(battery.max_act, normalized=False)

        self.assertEqual(info['provided_energy'], true_efficiency * DEFAULT_PARAMS['max_discharge'])
