import pytest

from tests.helpers.test_case import TestCase

from pymgrid import Microgrid
from pymgrid.algos import RuleBasedControl
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule
from pymgrid.forecast import OracleForecaster, GaussianNoiseForecaster


@pytest.mark.slow
class TestRBCScenario0(TestCase):
    microgrid_number = 0

    def setUp(self) -> None:
        microgrid = Microgrid.from_scenario(microgrid_number=self.microgrid_number)
        self.rbc = RuleBasedControl(microgrid)

    def test_correct_forecasts_oracle_forecaster(self):
        self.rbc.microgrid.set_forecaster(forecaster='oracle', forecast_horizon=23)
        self.rbc.run()

        for module in self.rbc.microgrid.modules.iterlist():
            if not isinstance(module, BaseTimeSeriesMicrogridModule):
                continue

            self.assertIsInstance(module.forecaster, OracleForecaster)

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
                        self.assertEqual(
                            current_value_log.iloc[forecast_step+1:],
                            shifted_forecast.iloc[forecast_step+1:]
                        )

    def test_correct_forecasts_gaussian_forecaster_zero_noise(self):
        self.rbc.microgrid.set_forecaster(forecaster=0.0, forecast_horizon=5)
        self.rbc.run()

        for module in self.rbc.microgrid.modules.iterlist():
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
                        self.assertEqual(
                            current_value_log.iloc[forecast_step+1:],
                            shifted_forecast.iloc[forecast_step+1:]
                        )


class TestRBCScenario1(TestRBCScenario0):
    microgrid_number = 1


class TestRBCScenario2(TestRBCScenario0):
    microgrid_number = 2


class TestRBCScenario3(TestRBCScenario0):
    microgrid_number = 3


class TestRBCScenario4(TestRBCScenario0):
    microgrid_number = 4


class TestRBCScenario5(TestRBCScenario0):
    microgrid_number = 5


class TestRBCScenario6(TestRBCScenario0):
    microgrid_number = 6


class TestRBCScenario47(TestRBCScenario0):
    microgrid_number = 7


class TestRBCScenario8(TestRBCScenario0):
    microgrid_number = 8


class TestRBCScenario9(TestRBCScenario0):
    microgrid_number = 9


class TestRBCScenario10(TestRBCScenario0):
    microgrid_number = 10


class TestRBCScenario11(TestRBCScenario0):
    microgrid_number = 11


class TestRBCScenario12(TestRBCScenario0):
    microgrid_number = 12


class TestRBCScenario13(TestRBCScenario0):
    microgrid_number = 13


class TestRBCScenario14(TestRBCScenario0):
    microgrid_number = 14


class TestRBCScenario15(TestRBCScenario0):
    microgrid_number = 15


class TestRBCScenario16(TestRBCScenario0):
    microgrid_number = 16


class TestRBCScenario17(TestRBCScenario0):
    microgrid_number = 17


class TestRBCScenario18(TestRBCScenario0):
    microgrid_number = 18


class TestRBCScenario19(TestRBCScenario0):
    microgrid_number = 19


class TestRBCScenario20(TestRBCScenario0):
    microgrid_number = 20


class TestRBCScenario21(TestRBCScenario0):
    microgrid_number = 21


class TestRBCScenario22(TestRBCScenario0):
    microgrid_number = 22


class TestRBCScenario23(TestRBCScenario0):
    microgrid_number = 23


class TestRBCScenario24(TestRBCScenario0):
    microgrid_number = 24
