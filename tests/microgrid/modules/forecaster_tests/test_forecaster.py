import numpy as np
from tests.helpers.test_case import TestCase
from pymgrid.forecast import (
    get_forecaster, OracleForecaster, GaussianNoiseForecaster, UserDefinedForecaster, NoForecaster)
from pymgrid.utils.space import ModuleSpace

STATE_COMPONENTS = np.random.randint(low=1, high=10)
FORECAST_HORIZON = np.random.randint(low=2, high=10)

POSITIVE_OBSERVATION_SPACE = ModuleSpace(
        unnormalized_low=0,
        unnormalized_high=10,
        shape=(STATE_COMPONENTS*(FORECAST_HORIZON+1),)
    )

NEGATIVE_OBSERVATION_SPACE = ModuleSpace(
        unnormalized_low=-10,
        unnormalized_high=0,
        shape=(STATE_COMPONENTS*(FORECAST_HORIZON+1),)
    )


def get_test_inputs(n=None, state_components=None, negative=False):
    state_components = state_components if state_components else STATE_COMPONENTS
    n = n if n else FORECAST_HORIZON
    val_c_n = POSITIVE_OBSERVATION_SPACE.unnormalized.high[0] * np.random.rand(n, state_components)
    val_c = val_c_n[0, :]
    # val_c_n = val_c_n.reshape((FORECAST_HORIZON, STATE_COMPONENTS))
    if negative:
        return -val_c, -val_c_n, n
    else:
        return val_c, val_c_n, n


class TestOracleForecaster(TestCase):
    def setUp(self) -> None:
        self.forecaster = OracleForecaster(observation_space=POSITIVE_OBSERVATION_SPACE,
                                           forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                           sink_only=False)

    def test_positive_inputs(self):
        val_c, val_c_n, n = get_test_inputs()
        forecast = self.forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_negative_inputs(self):
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = self.forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)


class TestGaussianNoiseForecaster(TestCase):
    def test_single_forecast_positive(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=POSITIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        val_c, val_c_n, n = get_test_inputs()
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast >= 0).all())

    def test_single_forecast_positive_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=POSITIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        val_c, val_c_n, n = get_test_inputs()
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast >= 0).all())

    def test_single_forecast_negative(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space= NEGATIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast <= 0).all())

    def test_single_forecast_negative_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=NEGATIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast <= 0).all())

    def test_multiple_forecast_positive(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=POSITIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast >= 0).all())

    def test_multiple_forecast_positive_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=POSITIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast >= 0).all())

    def test_multiple_forecast_negative(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=NEGATIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n, negative=True)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast <= 0).all())

    def test_multiple_forecast_negative_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=NEGATIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n, negative=True)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast <= 0).all())

    def test_increasing_uncertainty_positive(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=True,
                                             observation_space=POSITIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)
        val_c, val_c_n, n = get_test_inputs()
        expected_noise_std = np.outer(noise_std*(1+np.log(1+np.arange(len(val_c_n)))), np.ones(STATE_COMPONENTS))

        forecast = forecaster(val_c, val_c_n, n)
        self.assertTrue((noise_std != forecaster.noise_std).any())
        self.assertEqual(expected_noise_std, forecaster.noise_std)
        self.assertTrue((forecast >= 0).all())

    def test_increasing_uncertainty_negative(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=True,
                                             observation_space=NEGATIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)

        val_c, val_c_n, n = get_test_inputs(negative=True)
        expected_noise_std = np.outer(noise_std*(1+np.log(1+np.arange(len(val_c_n)))), np.ones(STATE_COMPONENTS))

        forecast = forecaster(val_c, val_c_n, n)
        self.assertTrue((noise_std != forecaster.noise_std).any())
        self.assertEqual(expected_noise_std, forecaster.noise_std)
        self.assertTrue((forecast <= 0).all())

    def test_bad_shape(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std,
                                             increase_uncertainty=False,
                                             observation_space=POSITIVE_OBSERVATION_SPACE,
                                             forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                             sink_only=False)

        n = np.random.randint(FORECAST_HORIZON, 2*FORECAST_HORIZON)
        val_c, val_c_n, n = get_test_inputs(n=n)
        with self.assertRaises(RuntimeError):
            _ = forecaster(val_c, val_c_n, n)
        # self.assertEqual(noise_std, forecaster.noise_std)
        # val_c, val_c_n, n = get_test_inputs(n=n_vals[1])
        # with self.assertRaises(ValueError):
        #     _ = forecaster(val_c, val_c_n, n)


class TestUserDefinedForecaster(TestCase):
    def setUp(self) -> None:
        self.simple_time_series = np.arange(FORECAST_HORIZON).reshape((-1, 1))

    @staticmethod
    def oracle_scalar_forecaster(val_c, val_c_n, n):
        return val_c_n.item()

    def get_oracle_forecaster(self, negative=False):
        return OracleForecaster(observation_space=self.get_obs_space(negative=negative),
                                forecast_shape=(FORECAST_HORIZON,),
                                sink_only=False)

    def get_obs_space(self, negative=False):
        if negative:
            low = -10
            high = 0
        else:
            low = 0
            high = 10

        return ModuleSpace(unnormalized_low=low, unnormalized_high=high, shape=(10, ))

    def test_user_defined_oracle_positive(self):
        forecaster = UserDefinedForecaster(forecaster_function=self.get_oracle_forecaster(),
                                           time_series=self.simple_time_series,
                                           observation_space=self.get_obs_space(),
                                           forecast_shape=(FORECAST_HORIZON, ),
                                           sink_only=False)
        val_c, val_c_n, n = get_test_inputs(state_components=1)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_user_defined_oracle_negative(self):
        forecaster = UserDefinedForecaster(forecaster_function=self.get_oracle_forecaster(negative=True),
                                           time_series=self.simple_time_series,
                                           observation_space=self.get_obs_space(negative=True),
                                           forecast_shape=(FORECAST_HORIZON,),
                                           sink_only=False)
        val_c, val_c_n, n = get_test_inputs(state_components=1, negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_scalar_forecaster(self):
        forecaster = UserDefinedForecaster(forecaster_function=self.oracle_scalar_forecaster,
                                           time_series=self.simple_time_series,
                                           observation_space=self.get_obs_space(negative=False),
                                           forecast_shape=(FORECAST_HORIZON,),
                                           sink_only=False)
        val_c, val_c_n, n = get_test_inputs(state_components=1, negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_vectorized_forecaster_bad_output_shape(self):
        bad_output_shape_forecaster = lambda val_c, val_c_n, n: np.append(val_c_n, [0])
        with self.assertRaisesRegex(ValueError, "Forecaster output of shape (.*) "
                                                "cannot be casted to necessary forecast shape (.*, 1)"):
            _ = UserDefinedForecaster(forecaster_function=bad_output_shape_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=POSITIVE_OBSERVATION_SPACE,
                                      forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                      sink_only=False)

    def test_vectorized_forecaster_bad_output_type(self):
        bad_output_type_forecaster = lambda val_c, val_c_n, n: np.array([str(x) for x in val_c_n]).reshape((-1, 1))
        with self.assertRaisesRegex(TypeError, "Forecaster must return numeric np.ndarray or number but returned "
                                               "output of type"):
            _ = UserDefinedForecaster(forecaster_function=bad_output_type_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=POSITIVE_OBSERVATION_SPACE,
                                      forecast_shape=(FORECAST_HORIZON,),
                                      sink_only=False)

    def test_vectorized_forecaster_bad_output_signs(self):
        def bad_output_type_forecaster(val_c, val_c_n, n):
            out = val_c_n.copy()
            pos = np.random.randint(low=1, high=len(out))
            out[pos] *= -1
            return out

        with self.assertRaisesRegex(ValueError, "Forecaster must return output of same "
                                                "sign \(or zero\) as input but returned output"):
            _ = UserDefinedForecaster(forecaster_function=bad_output_type_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=POSITIVE_OBSERVATION_SPACE,
                                      forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS), sink_only=False)

    def test_bad_forecaster(self):
        bad_forecaster = lambda val_c, val_c_n, n: 0/0

        with self.assertRaisesRegex(ValueError, "Unable to call forecaster with scalar inputs."):
            _ = UserDefinedForecaster(forecaster_function=bad_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=POSITIVE_OBSERVATION_SPACE,
                                      forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                      sink_only=False)

    def test_scalar_forecaster_bad_output_shape(self):
        def bad_output_shape_forecaster(val_c, val_c_n, n):
            if hasattr(val_c_n, '__len__') and len(val_c_n) > 1:
                raise RuntimeError
            return [val_c_n]*2

        with self.assertRaisesRegex(ValueError, "Forecaster must return scalar output with scalar input but returned."):
            _ = UserDefinedForecaster(forecaster_function=bad_output_shape_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=POSITIVE_OBSERVATION_SPACE,
                                      forecast_shape=(FORECAST_HORIZON, STATE_COMPONENTS),
                                      sink_only=False)


class TestGetForecaster(TestCase):
    def setUp(self) -> None:
        self.simple_time_series = np.arange(10).reshape((-1, 1))
        self.forecaster_horizon = 24

    def test_user_defined_forecaster(self):
        user_defined_forecaster = lambda val_c, val_c_n, n: val_c_n
        forecaster = get_forecaster(user_defined_forecaster,
                                    POSITIVE_OBSERVATION_SPACE,
                                    (FORECAST_HORIZON, STATE_COMPONENTS),
                                    time_series=self.simple_time_series)
        self.assertIsInstance(forecaster, UserDefinedForecaster)

    def test_oracle_forecaster(self):
        forecaster = get_forecaster("oracle", POSITIVE_OBSERVATION_SPACE, (FORECAST_HORIZON, STATE_COMPONENTS))
        self.assertIsInstance(forecaster, OracleForecaster)

    def test_no_forecaster(self):
        forecaster = get_forecaster(None, POSITIVE_OBSERVATION_SPACE, (FORECAST_HORIZON, STATE_COMPONENTS))
        self.assertIsInstance(forecaster, NoForecaster)

    def test_gaussian_noise_forecaster(self):
        noise_std = 0.5
        forecaster = get_forecaster(noise_std, POSITIVE_OBSERVATION_SPACE, (FORECAST_HORIZON, STATE_COMPONENTS))
        self.assertIsInstance(forecaster, GaussianNoiseForecaster)
        self.assertEqual(forecaster.input_noise_std, noise_std)

    def test_gaussian_noise_forecaster_increase_uncertainty(self):
        noise_std = 0.5
        forecaster = get_forecaster(noise_std, POSITIVE_OBSERVATION_SPACE, (FORECAST_HORIZON, STATE_COMPONENTS), increase_uncertainty=True)
        self.assertIsInstance(forecaster, GaussianNoiseForecaster)
        self.assertEqual(forecaster.input_noise_std, noise_std)
        self.assertTrue((forecaster.noise_std != noise_std).any())
