import numpy as np
from tests.helpers.test_case import TestCase
from pymgrid.forecast import (
    get_forecaster, OracleForecaster, GaussianNoiseForecaster, UserDefinedForecaster, NoForecaster)
from pymgrid.utils.space import ModuleSpace

OBSERVATION_SPACE = ModuleSpace(unnormalized_low=0, unnormalized_high=10, shape=(10,))

def get_test_inputs(n=None, negative=False):
    n = n if n else np.random.randint(low=2, high=OBSERVATION_SPACE.shape[0])
    val_c_n = OBSERVATION_SPACE.unnormalized.high[0] * np.random.rand(n)
    val_c = val_c_n[0]
    val_c_n = val_c_n.reshape((-1, 1))
    if negative:
        return -val_c, -val_c_n, n
    else:
        return val_c, val_c_n, n


class TestOracleForecaster(TestCase):
    def setUp(self) -> None:
        self.forecaster = OracleForecaster(observation_space=OBSERVATION_SPACE)

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
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs()
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast >= 0).all())

    def test_single_forecast_positive_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs()
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast >= 0).all())

    def test_single_forecast_negative(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast <= 0).all())

    def test_single_forecast_negative_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        self.assertTrue((forecast <= 0).all())

    def test_multiple_forecast_positive(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast >= 0).all())

    def test_multiple_forecast_positive_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast >= 0).all())

    def test_multiple_forecast_negative(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n, negative=True)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast <= 0).all())

    def test_multiple_forecast_negative_high_std(self):
        noise_std = 100
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        n = None
        for _ in range(2):
            val_c, val_c_n, n = get_test_inputs(n=n, negative=True)
            forecast = forecaster(val_c, val_c_n, n)
            self.assertEqual(noise_std, forecaster.noise_std)
            self.assertTrue((forecast <= 0).all())

    def test_increasing_uncertainty_positive(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=True, observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs()
        forecast = forecaster(val_c, val_c_n, n)
        self.assertTrue((noise_std != forecaster.noise_std).any())
        self.assertEqual(noise_std*(1+np.log(1+np.arange(len(val_c_n)))), forecaster.noise_std)
        self.assertTrue((forecast >= 0).all())

    def test_increasing_uncertainty_negative(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=True, observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertTrue((noise_std != forecaster.noise_std).any())
        self.assertEqual(noise_std*(1+np.log(1+np.arange(len(val_c_n)))), forecaster.noise_std)
        self.assertTrue((forecast <= 0).all())

    def test_bad_shape(self):
        noise_std = 1
        forecaster = GaussianNoiseForecaster(noise_std=noise_std, increase_uncertainty=False, observation_space=OBSERVATION_SPACE)
        n_vals = [5, 10]
        val_c, val_c_n, n = get_test_inputs(n=n_vals[0])
        _ = forecaster(val_c, val_c_n, n)
        self.assertEqual(noise_std, forecaster.noise_std)
        val_c, val_c_n, n = get_test_inputs(n=n_vals[1])
        with self.assertRaises(ValueError):
            _ = forecaster(val_c, val_c_n, n)


class TestUserDefinedForecaster(TestCase):
    def setUp(self) -> None:
        self.oracle_forecaster = OracleForecaster(observation_space=OBSERVATION_SPACE)
        self.simple_time_series = np.arange(10).reshape((-1, 1))

    @staticmethod
    def oracle_scalar_forecaster(val_c, val_c_n, n):
        return val_c_n.item()

    def test_user_defined_oracle_positive(self):
        forecaster = UserDefinedForecaster(forecaster_function=self.oracle_forecaster,
                                           time_series=self.simple_time_series,
                                           observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs()
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_user_defined_oracle_negative(self):
        forecaster = UserDefinedForecaster(forecaster_function=self.oracle_forecaster,
                                           time_series=self.simple_time_series,
                                           observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_scalar_forecaster(self):
        forecaster = UserDefinedForecaster(forecaster_function=self.oracle_scalar_forecaster,
                                           time_series=self.simple_time_series,
                                           observation_space=OBSERVATION_SPACE)
        val_c, val_c_n, n = get_test_inputs(negative=True)
        forecast = forecaster(val_c, val_c_n, n)
        self.assertEqual(forecast, val_c_n)

    def test_vectorized_forecaster_bad_output_shape(self):
        bad_output_shape_forecaster = lambda val_c, val_c_n, n: np.append(val_c_n, [0])
        with self.assertRaisesRegex(ValueError, "Forecaster output of shape (.*) "
                                                "cannot be casted to necessary forecast shape (.*, 1)"):
            _ = UserDefinedForecaster(forecaster_function=bad_output_shape_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=OBSERVATION_SPACE)

    def test_vectorized_forecaster_bad_output_type(self):
        bad_output_type_forecaster = lambda val_c, val_c_n, n: np.array([str(x) for x in val_c_n]).reshape((-1, 1))
        with self.assertRaisesRegex(TypeError, "Forecaster must return numeric np.ndarray or number but returned "
                                               "output of type"):
            _ = UserDefinedForecaster(forecaster_function=bad_output_type_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=OBSERVATION_SPACE)

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
                                      observation_space=OBSERVATION_SPACE)

    def test_bad_forecaster(self):
        bad_forecaster = lambda val_c, val_c_n, n: 0/0

        with self.assertRaisesRegex(ValueError, "Unable to call forecaster with scalar inputs."):
            _ = UserDefinedForecaster(forecaster_function=bad_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=OBSERVATION_SPACE)

    def test_scalar_forecaster_bad_output_shape(self):
        def bad_output_shape_forecaster(val_c, val_c_n, n):
            if hasattr(val_c_n, '__len__') and len(val_c_n) > 1:
                raise RuntimeError
            return [val_c_n]*2

        with self.assertRaisesRegex(ValueError, "Forecaster must return scalar output with scalar input but returned."):
            _ = UserDefinedForecaster(forecaster_function=bad_output_shape_forecaster,
                                      time_series=self.simple_time_series,
                                      observation_space=OBSERVATION_SPACE)


class TestGetForecaster(TestCase):
    def setUp(self) -> None:
        self.simple_time_series = np.arange(10).reshape((-1, 1))
        self.forecaster_horizon = 24

    def test_user_defined_forecaster(self):
        user_defined_forecaster = lambda val_c, val_c_n, n: val_c_n
        forecaster = get_forecaster(user_defined_forecaster,
                                                        self.forecaster_horizon,
                                                        OBSERVATION_SPACE,
                                                        self.simple_time_series)
        self.assertIsInstance(forecaster, UserDefinedForecaster)

    def test_oracle_forecaster(self):
        forecaster = get_forecaster("oracle", self.forecaster_horizon, OBSERVATION_SPACE)
        self.assertIsInstance(forecaster, OracleForecaster)

    def test_no_forecaster(self):
        forecaster = get_forecaster(None, self.forecaster_horizon, OBSERVATION_SPACE)
        self.assertIsInstance(forecaster, NoForecaster)

    def test_gaussian_noise_forecaster(self):
        noise_std = 0.5
        forecaster = get_forecaster(noise_std, self.forecaster_horizon, OBSERVATION_SPACE)
        self.assertIsInstance(forecaster, GaussianNoiseForecaster)
        self.assertEqual(forecaster.input_noise_std, noise_std)

    def test_gaussian_noise_forecaster_increase_uncertainty(self):
        noise_std = 0.5
        forecaster = get_forecaster(noise_std, self.forecaster_horizon, OBSERVATION_SPACE,
                                                        increase_uncertainty=True)
        self.assertIsInstance(forecaster, GaussianNoiseForecaster)
        self.assertEqual(forecaster.input_noise_std, noise_std)

        with self.assertRaisesRegex(TypeError, "unsupported operand type\(s\)"):
            _ = forecaster.noise_std
