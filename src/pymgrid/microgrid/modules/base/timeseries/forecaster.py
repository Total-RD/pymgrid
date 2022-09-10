import numpy as np
from pandas.api.types import is_number
from abc import abstractmethod


def get_forecaster(forecaster, time_series):
    """
    Get the forecasting function for the time series module.
    :param forecaster: callable, float, or "oracle" default "oracle". Function that gives a forecast n-steps ahead.
        If callable, must take as arguments (val_c: float, val_{c+n}: float, n: int), where:
            val_c is the current value in the time series: self.time_series[self.current_step],
            val_{c+n} is the value in the time series n steps in the future,
            n is the number of steps in the future at which we are forecasting.
            The output forecast = forecaster(val_c, val_{c+n}, n) must have the same sign
            as the inputs val_c and val_{c+n}.

        If float, serves as a standard deviation for a mean-zero gaussian noise function
            that is added to the true value.

        If "oracle", gives a perfect forecast.

    :return:
    forecast, callable[float, float, int]. The forecasting function.
    """
    if callable(forecaster):
        return UserDefinedForecaster(forecaster, time_series)
    if forecaster == "oracle":
        return OracleForecaster()
    elif is_number(forecaster):
        return GaussianNoiseForecaster(forecaster)


class Forecaster:
    @abstractmethod
    def __call__(self, val_c, val_c_n, n):
        pass


class UserDefinedForecaster(Forecaster):
    def __init__(self, forecaster, time_series):
        is_vectorized_forecaster = _validate_callable_forecaster(forecaster, time_series)
        if not is_vectorized_forecaster:
            forecaster = vectorize_scalar_forecaster(forecaster)
        self._forecaster = forecaster

    def __call__(self, val_c, val_c_n, n):
        return self._forecaster(val_c, val_c_n, n)


class OracleForecaster(Forecaster):
    def __call__(self, val_c, val_c_n, n):
        return val_c_n


class GaussianNoiseForecaster(Forecaster):
    def __init__(self, noise_std, increase_uncertainty=False):
        self.input_noise_std = noise_std
        self.increase_uncertainty=increase_uncertainty
        self._noise_size = None
        self._noise_std = None

    def _get_noise_std(self):
        if self.increase_uncertainty:
            return self.input_noise_std+(1+np.log(np.arange(self._noise_size)))
        else:
            return self.input_noise_std

    def _get_noise(self, size):
        if not self._noise_size:
            self._noise_size = size
        if size != self._noise_size:
            raise ValueError(f"size {size} incompatible with previous size {self._noise_size}")
        return np.random.normal(scale=self.noise_std, size=size)

    @property
    def noise_std(self):
        if not self._noise_std:
            self._noise_std = self._get_noise_std()
        return self._noise_std

    def __call__(self, val_c, val_c_n, n):
        forecast =  val_c_n + self._get_noise(val_c_n.shape)
        forecast[(forecast*val_c_n) < 0] = 0
        return forecast


def _validate_callable_forecaster(forecaster, time_series):
    val_c = time_series[0]
    n = np.random.randint(2, len(time_series))
    vector_true_forecast = time_series[:n]
    try:
        _validate_vectorized_forecaster(forecaster, val_c, vector_true_forecast, n)
        is_vectorized_forecaster = True
    except Exception:
        scalar_true_forecast = vector_true_forecast[-1]
        _validate_scalar_forecaster(forecaster, val_c, scalar_true_forecast, n)
        is_vectorized_forecaster = False
    return is_vectorized_forecaster


def _validate_vectorized_forecaster(forecaster, val_c, vector_true_forecast, n):
    try:
        vectorized_forecast = forecaster(val_c, vector_true_forecast, n)
    except Exception as e:
        raise ValueError("Unable to call forecaster with vector inputs."
                         f"\nFunc call forecaster(val_c={val_c}, val_c_n={vector_true_forecast}, n={n})"
                         f"\nraised {e}") from e
    else:
        # vectorized function call succeeded
        if vectorized_forecast.shape != vector_true_forecast.shape:
            raise ValueError(f"Forecaster vectorized output with shape {vectorized_forecast.shape}"
                             f"does not match input shape {vector_true_forecast.shape}")

        for i, (forecast, true_forecast) in enumerate(zip(vectorized_forecast, vector_true_forecast)):
            try:
                _validate_forecasted_value(forecast, true_forecast, val_c, n)
            except Exception as e:
                raise ValueError(f"Failed validating forecast at position {i} due to exception {e}") from e


def _validate_scalar_forecaster(forecaster, val_c, scalar_true_forecast, n):
    try:
        scalar_forecast = forecaster(val_c, scalar_true_forecast, n)
    except Exception as e_scalar:
        raise ValueError("Unable to call forecaster with scalar inputs."
                         f"\nFunc call forecaster(val_c={val_c}, val_c_plus_n={scalar_true_forecast}, n={n})"
                         f"\nraised {e_scalar}") from e_scalar
    else:  # scalar function call succeeded
        # check shape
        try:
            scalar_forecast_item = scalar_forecast.item()
        except AttributeError:
            pass
        except ValueError:
            raise ValueError("Unable to validate forecaster. Forecaster must return scalar output with scalar"
                             f"input but returned {scalar_forecast}")
        else:
            _validate_forecasted_value(scalar_forecast_item, scalar_true_forecast, val_c, n)


def _validate_forecasted_value(forecaster_output, true_forecast, val_c, n):
    if not is_number(forecaster_output):
        raise ValueError(
            "Unable to validate forecaster. Forecaster must return numeric output but returned"
            f"output of type {type(forecaster_output)}: {forecaster_output}")
    elif not (forecaster_output * true_forecast >= 0):
        raise ValueError(
            "Unable to validate forecaster. Forecaster must return output of same sign (or zero) as"
            f"input but returned output {forecaster_output} with inputs"
            f"val_c={val_c}, val_c_plus_n={true_forecast}, n={n}")


def vectorize_scalar_forecaster(forecaster):
    def vectorized(val_c, val_c_n, n):
        if n != len(val_c_n):
            raise ValueError(f"Incompatible true values length ({val_c_n}) to forecast {n}-steps ahead.")
        return np.array([forecaster(val_c, val_c_n_i, n_i) for n_i, val_c_n_i in enumerate(val_c_n)])
    return vectorized
