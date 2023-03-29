import pytest


from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid import Microgrid
from pymgrid.algos import ModelPredictiveControl
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule
from pymgrid.forecast import OracleForecaster, GaussianNoiseForecaster


@pytest.mark.slow
class MPCScenario(TestCase):
    microgrid_number: int

    def setUp(self) -> None:
        microgrid = Microgrid.from_scenario(microgrid_number=self.microgrid_number)
        self.mpc = ModelPredictiveControl(microgrid)

    def test_correct_forecasts_oracle_forecaster(self):
        self.mpc.microgrid.set_forecaster(forecaster='oracle', forecast_horizon=23)
        self.mpc.run()

        for module in self.mpc.microgrid.modules.iterlist():
            if not isinstance(module, BaseTimeSeriesMicrogridModule):
                continue

            self.assertIsInstance(module.forecaster, OracleForecaster)

            for state_component in module.state_components:
                current_value_log = module.log[f'{state_component}_current']

                for forecast_step in range(module.forecast_horizon):
                    forecast_value_log = module.log[f'{state_component}_forecast_{forecast_step}']
                    shifted_forecast = forecast_value_log.shift(forecast_step+1)

                    with self.subTest(
                            module_name=module.name,
                            state_component=state_component,
                            forecast_step=forecast_step
                    ):
                        self.assertEqual(current_value_log.iloc[forecast_step:], shifted_forecast.iloc[forecast_step:])

        self.assertTrue(False)

    def test_correct_forecasts_gaussian_forecaster_zero_noise(self):
        self.microgrid.set_forecaster(forecaster=0.0, forecast_horizon=23)
        self.mpc.run()

        for module in self.microgrid.modules.iterlist():
            if not isinstance(module, BaseTimeSeriesMicrogridModule):
                continue

            self.assertIsInstance(module.forecaster, GaussianNoiseForecaster)

            for state_component in module.state_components:
                current_value_log = module.log[f'{state_component}_current']

                for forecast_step in range(module.forecast_horizon):
                    forecast_value_log = module.log[f'{state_component}_forecast_{forecast_step}']
                    shifted_forecast = forecast_value_log.shift(forecast_step + 1)

                    with self.subTest(
                            module_name=module.name,
                            state_component=state_component,
                            forecast_step=forecast_step
                    ):
                        self.assertEqual(current_value_log.iloc[forecast_step:], shifted_forecast.iloc[forecast_step:])

        self.assertTrue(False)


class TestMPCScenario0(MPCScenario):
    microgrid_number = 1


class TestMPCScenario1(MPCScenario):
    microgrid_number = 1


class TestMPCScenario2(MPCScenario):
    microgrid_number = 2


class TestMPCScenario3(MPCScenario):
    microgrid_number = 3


class TestMPCScenario4(MPCScenario):
    microgrid_number = 4


class TestMPCScenario5(MPCScenario):
    microgrid_number = 5


class TestMPCScenario6(MPCScenario):
    microgrid_number = 6


class TestMPCScenario47(MPCScenario):
    microgrid_number = 7


class TestMPCScenario8(MPCScenario):
    microgrid_number = 8


class TestMPCScenario9(MPCScenario):
    microgrid_number = 9


class TestMPCScenario10(MPCScenario):
    microgrid_number = 10


class TestMPCScenario11(MPCScenario):
    microgrid_number = 11


class TestMPCScenario12(MPCScenario):
    microgrid_number = 12


class TestMPCScenario13(MPCScenario):
    microgrid_number = 13


class TestMPCScenario14(MPCScenario):
    microgrid_number = 14


class TestMPCScenario15(MPCScenario):
    microgrid_number = 15


class TestMPCScenario16(MPCScenario):
    microgrid_number = 16


class TestMPCScenario17(MPCScenario):
    microgrid_number = 17


class TestMPCScenario18(MPCScenario):
    microgrid_number = 18


class TestMPCScenario19(MPCScenario):
    microgrid_number = 19


class TestMPCScenario20(MPCScenario):
    microgrid_number = 20


class TestMPCScenario21(MPCScenario):
    microgrid_number = 21


class TestMPCScenario22(MPCScenario):
    microgrid_number = 22


class TestMPCScenario23(MPCScenario):
    microgrid_number = 23


class TestMPCScenario24(MPCScenario):
    microgrid_number = 24
