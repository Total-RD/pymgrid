import numpy as np
from pandas.api.types import is_number, is_numeric_dtype
from abc import abstractmethod


def get_forecaster(forecaster, forecast_horizon, time_series=None, increase_uncertainty=False):
    """
    Get the forecasting function for the time series module.
    :param forecaster: callable, float, "oracle", or None, default None. Function that gives a forecast n-steps ahead.
        If callable, must take as arguments (val_c: float, val_{c+n}: float, n: int), where:
            val_c is the current value in the time series: self.time_series[self.current_step],
            val_{c+n} is the value in the time series n steps in the future,
            n is the number of steps in the future at which we are forecasting.
            The output forecast = forecaster(val_c, val_{c+n}, n) must have the same sign
            as the inputs val_c and val_{c+n}.

        If float, serves as a standard deviation for a mean-zero gaussian noise function
            that is added to the true value.

        If "oracle", gives a perfect forecast.

        If None, no forecast.

    :param forecast_horizon: int. Number of steps in the future to forecast. If forecaster is None, ignored and 0 is returned.

    :param time_series: ndarray[float] or None, default None.
        The underlying time series, used to validate UserDefinedForecaster.
        Only used if callable(forecaster).

    :param increase_uncertainty: bool, default False. Whether to increase uncertainty for farther-out dates if using
        a GaussianNoiseForecaster. Ignored otherwise.

    :return:
    forecast, callable[float, float, int]. The forecasting function.
    """

    if forecaster is None:
        return NoForecaster(), 0
    elif isinstance(forecaster, (UserDefinedForecaster, OracleForecaster, GaussianNoiseForecaster)):
        return forecaster, forecast_horizon
    elif callable(forecaster):
        return UserDefinedForecaster(forecaster, time_series), forecast_horizon
    elif forecaster == "oracle":
        return OracleForecaster(), forecast_horizon
    elif is_number(forecaster):
        return GaussianNoiseForecaster(forecaster, increase_uncertainty=increase_uncertainty), forecast_horizon
    else:
        raise ValueError(f"Unable to parse forecaster of type {type(forecaster)}")


class Forecaster:
    @abstractmethod
    def _forecast(self, val_c, val_c_n, n):
        pass

    def _pad(self, forecast, n):
        if forecast.shape[0] == n:
            return forecast
        else:
            pad_amount = n-forecast.shape[0]
            return np.pad(forecast, ((0, pad_amount), (0, 0)), constant_values=0)

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return self.__dict__ == other.__dict__

    def __call__(self, val_c, val_c_n, n):
        if len(val_c_n.shape) == 1:
            val_c_n = val_c_n.reshape((-1, 1))
        forecast = self._forecast(val_c, val_c_n, n)

        if forecast is None:
            return None
        else:
            forecast = self._pad(forecast, n)
            assert forecast.shape == val_c_n.shape
            return forecast

    def __repr__(self):
        return self.__class__.__name__


class UserDefinedForecaster(Forecaster):
    def __init__(self, forecaster_function, time_series):
        self.is_vectorized_forecaster, self.cast_to_arr = \
            _validate_callable_forecaster(forecaster_function, time_series)

        if not self.is_vectorized_forecaster:
            forecaster_function = vectorize_scalar_forecaster(forecaster_function)

        self._forecaster = forecaster_function

    def _cast_to_arr(self, forecast, val_c_n):
        if self.cast_to_arr:
            return np.array(forecast.reshape(val_c_n.shape))
        return forecast

    def _forecast(self, val_c, val_c_n, n):
        forecast = self._forecaster(val_c, val_c_n, n)
        return self._cast_to_arr(forecast, val_c_n)


class OracleForecaster(Forecaster):
    def _forecast(self, val_c, val_c_n, n):
        return val_c_n


class GaussianNoiseForecaster(Forecaster):
    def __init__(self, noise_std, increase_uncertainty=False):
        self.input_noise_std = noise_std
        self.increase_uncertainty = increase_uncertainty
        self._noise_size = None
        self._noise_std = None

    def _get_noise_std(self):
        if self.increase_uncertainty:
            return self.input_noise_std*(1+np.log(1+np.arange(self._noise_size)))
        else:
            return self.input_noise_std

    def _get_noise(self, size):
        if self._noise_size is None:
            self._noise_size = size
        if size != self._noise_size:
            raise ValueError(f"size {size} incompatible with previous size {self._noise_size}")
        return np.random.normal(scale=self.noise_std, size=size)

    @property
    def noise_std(self):
        if self._noise_std is None:
            self._noise_std = self._get_noise_std()
        return self._noise_std

    def _forecast(self, val_c, val_c_n, n):
        forecast = val_c_n + self._get_noise(len(val_c_n)).reshape(val_c_n.shape)
        forecast[(forecast*val_c_n) < 0] = 0
        return forecast

    def __repr__(self):
        return f'GaussianNoiseForecaster(noise_std={self.input_noise_std}, ' \
               f'increase_uncertainty={self.increase_uncertainty}'


class NoForecaster(Forecaster):
    def _forecast(self, val_c, val_c_n, n):
        return None


def _validate_callable_forecaster(forecaster, time_series):
    val_c = time_series[0]
    n = np.random.randint(2, len(time_series))
    vector_true_forecast = time_series[:n]
    try:
        cast_to_arr = _validate_vectorized_forecaster(forecaster, val_c, vector_true_forecast, n)
        is_vectorized_forecaster = True
    except NotImplementedError:
        scalar_true_forecast = vector_true_forecast[-1]
        _validate_scalar_forecaster(forecaster, val_c, scalar_true_forecast, n)
        is_vectorized_forecaster = False
        cast_to_arr = False

    return is_vectorized_forecaster, cast_to_arr


def _validate_vectorized_forecaster(forecaster, val_c, vector_true_forecast, n):
    try:
        vectorized_forecast = forecaster(val_c, vector_true_forecast, n)
    except Exception as e:
        raise NotImplementedError("Unable to call forecaster with vector inputs. "
                         f"\nFunc call forecaster(val_c={val_c}, val_c_n={vector_true_forecast}, n={n})"
                         f"\nraised {type(e).__name__}: {e}") from e
    else:
        # vectorized function call succeeded
        if not hasattr(vectorized_forecast, 'size'):
            vectorized_forecast = np.array(vectorized_forecast)
            cast_to_arr = True
        else:
            cast_to_arr = False
        try:
            vectorized_forecast = vectorized_forecast.reshape(vector_true_forecast.shape)
        except ValueError:
            raise ValueError(f"Forecaster output of shape {vectorized_forecast.shape} cannot be casted to "
                             f"necessary forecast shape {vector_true_forecast.shape}")

        for i, (forecast, true_forecast) in enumerate(zip(vectorized_forecast, vector_true_forecast)):
            try:
                _validate_forecasted_value(forecast, true_forecast, val_c, n)
            except Exception as e:
                raise type(e)(f"Failed validating forecast at position {i} due to exception {e}") from e

        return cast_to_arr


def _validate_scalar_forecaster(forecaster, val_c, scalar_true_forecast, n):
    try:
        scalar_forecast = forecaster(val_c, scalar_true_forecast, n)
    except Exception as e_scalar:
        raise ValueError("Unable to call forecaster with scalar inputs. "
                         f"\nFunc call forecaster(val_c={val_c}, val_c_plus_n={scalar_true_forecast}, n={n})"
                         f"\nraised {type(e_scalar).__name__}: {e_scalar}") from e_scalar
    else:  # scalar function call succeeded
        # check shape
        try:
            assert is_number(scalar_forecast)
            scalar_forecast_item = scalar_forecast
        except AssertionError:
            try:
                scalar_forecast_item = scalar_forecast.item()
            except (ValueError, AttributeError):
                raise ValueError("Unable to validate forecaster. Forecaster must return scalar output with scalar "
                                f"input but returned {scalar_forecast}")

        _validate_forecasted_value(scalar_forecast_item, scalar_true_forecast, val_c, n)


def _validate_forecasted_value(forecaster_output, true_forecast, val_c, n):
    if not is_numeric_dtype(np.array(forecaster_output)):
        raise TypeError(
            "Unable to validate forecaster. Forecaster must return numeric np.ndarray or number but returned "
            f"output of type {np.array(forecaster_output).dtype}: {forecaster_output}")
    elif not (forecaster_output * true_forecast >= 0):
        raise ValueError(
            "Unable to validate forecaster. Forecaster must return output of same sign (or zero) as "
            f"input but returned output {forecaster_output} with inputs "
            f"val_c={val_c}, val_c_plus_n={true_forecast}, n={n}")


def vectorize_scalar_forecaster(forecaster):
    def vectorized(val_c, val_c_n, n):
        if n != len(val_c_n):
            raise ValueError(f"Incompatible true values length ({val_c_n}) to forecast {n}-steps ahead.")
        vectorized_output = np.array([forecaster(val_c, val_c_n_i, n_i) for n_i, val_c_n_i in enumerate(val_c_n)])
        try:
            shape = (-1, vectorized_output.shape[1])
        except IndexError:
            shape = (-1, 1)
        return vectorized_output.reshape(shape)
    return vectorized
