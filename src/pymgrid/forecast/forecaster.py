import numpy as np
from pandas.api.types import is_number, is_numeric_dtype
from abc import abstractmethod

from pymgrid.utils.space import ModuleSpace


def get_forecaster(forecaster, observation_space, forecast_shape, time_series=None, increase_uncertainty=False):
    """
    Get the forecasting function for the time series module.

    Parameters
    ----------
    forecaster : callable, float, "oracle", or None, default None.
        Function that gives a forecast n-steps ahead.

        * If ``callable``, must take as arguments ``(val_c: float, val_{c+n}: float, n: int)``, where

          * ``val_c`` is the current value in the time series: ``self.time_series[self.current_step]``

          * ``val_{c+n}`` is the value in the time series n steps in the future

          * n is the number of steps in the future at which we are forecasting.

          The output ``forecast = forecaster(val_c, val_{c+n}, n)`` must have the same sign
          as the inputs ``val_c`` and ``val_{c+n}``.

        * If ``float``, serves as a standard deviation for a mean-zero gaussian noise function
          that is added to the true value.

        * If ``"oracle"``, gives a perfect forecast.

        * If ``None``, no forecast.

    forecast_shape : int or tuple of int
        Expected shape of forecasts. If an integer, will return forecasts of shape (shape, 1).

    observation_space : :class:`ModuleSpace <pymgrid.utils.space.ModuleSpace>`
        Observation space; used to determine values to pad missing forecasts when we are forecasting past the
        end of the time series.

    time_series: ndarray[float] or None, default None.
        The underlying time series, used to validate UserDefinedForecaster.
        Only used if callable(forecaster).

    increase_uncertainty : bool, default False.
       Whether to increase uncertainty for farther-out dates if using
       a GaussianNoiseForecaster. Ignored otherwise.

    Returns
    -------
    forecaster : :class:`.Forecaster`
        The forecasting function.

    """

    if forecaster is None:
        return NoForecaster(observation_space, forecast_shape)
    elif isinstance(forecaster, (UserDefinedForecaster, OracleForecaster, GaussianNoiseForecaster)):
        return forecaster
    elif callable(forecaster):
        return UserDefinedForecaster(forecaster, observation_space, forecast_shape, time_series)
    elif forecaster == "oracle":
        return OracleForecaster(observation_space, forecast_shape)
    elif is_number(forecaster):
        return GaussianNoiseForecaster(forecaster, observation_space, forecast_shape,
                                       increase_uncertainty=increase_uncertainty)
    else:
        raise ValueError(f"Unable to parse forecaster of type {type(forecaster)}")


class Forecaster:
    def __init__(self, observation_space, forecast_shape):
        self._observation_space = observation_space
        self._forecast_shaped_space = self._get_forecast_shaped_space(forecast_shape)
        self._fill_arr = (self._observation_space.unnormalized.high + self._observation_space.unnormalized.low) / 2

    def _get_forecast_shaped_space(self, shape):
        if len(shape) == 1:
            shape = (*shape, 1)
        elif len(shape) > 2:
            raise ValueError(f'shape must be one- or two-dimensional.')

        n_in_forecast = shape[0]*shape[1]

        if n_in_forecast:
            unnormalized_low = self._observation_space.unnormalized.low[-n_in_forecast:]
            unnormalized_high = self._observation_space.unnormalized.high[-n_in_forecast:]
        else:
            unnormalized_low = np.array([])
            unnormalized_high = np.array([])

        return ModuleSpace(unnormalized_low=unnormalized_low.reshape(shape),
                           unnormalized_high=unnormalized_high.reshape(shape),
                           shape=shape)

    @abstractmethod
    def _forecast(self, val_c, val_c_n, n):
        pass

    def _pad(self, forecast, n):
        if forecast.shape[0] == n:
            return forecast
        else:
            pad_amount = n - forecast.shape[0]
            pad = self._fill_arr.reshape((-1, forecast.shape[1]))[-pad_amount:]

            if pad.shape[0] < pad_amount:
                raise RuntimeError("Attempting to pad a forecast to a value larger than the module's observation space "
                                   "implies.")

            return np.concatenate((forecast, pad))

    def full_pad(self, shape, forecast_horizon):
        if forecast_horizon is None:
            return None
        empty_forecast = np.array([]).reshape((0, shape[1]))
        return self._pad(empty_forecast, forecast_horizon)

    def _clip(self, forecast):
        lb = self._forecast_shaped_space.unnormalized.low[-forecast.shape[0]:]
        ub = self._forecast_shaped_space.unnormalized.high[-forecast.shape[0]:]
        lt_lb = forecast < lb
        gt_ub = forecast > ub

        forecast[lt_lb] = lb[lt_lb]
        forecast[gt_ub] = ub[gt_ub]

        return forecast

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value
        self._fill_arr = (self._observation_space.unnormalized.high + self._observation_space.unnormalized.low) / 2
        new_shape = (
            int((value.shape[0] - self._forecast_shaped_space.shape[1]) / self._forecast_shaped_space.shape[1]),
            self._forecast_shaped_space.shape[1]
        )
        self._forecast_shaped_space = self._get_forecast_shaped_space(new_shape)

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        return (self._fill_arr == other._fill_arr).all() and \
               all(v == other.__dict__[k] for k, v in self.__dict__.items() if k != '_fill_arr')

    def __call__(self, val_c, val_c_n, n):
        if len(val_c_n.shape) == 1:
            val_c_n = val_c_n.reshape((-1, 1))

        if val_c_n.shape[0] > self._forecast_shaped_space.shape[0]:
            raise RuntimeError(f'val_c_n shape {val_c_n.shape} is too large for space {self._forecast_shaped_space.shape}')

        forecast = self._forecast(val_c, val_c_n, n)

        if forecast is None:
            return None
        else:
            forecast = self._pad(forecast, n)
            forecast = self._clip(forecast)
            assert forecast.shape == (n, val_c_n.shape[1])
            return forecast

    def __repr__(self):
        return self.__class__.__name__


class UserDefinedForecaster(Forecaster):
    def __init__(self, forecaster_function, observation_space, forecast_shape, time_series):
        self.is_vectorized_forecaster, self.cast_to_arr = \
            _validate_callable_forecaster(forecaster_function, time_series)

        if not self.is_vectorized_forecaster:
            forecaster_function = vectorize_scalar_forecaster(forecaster_function)

        self._forecaster = forecaster_function

        super().__init__(observation_space, forecast_shape)

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
    def __init__(self, noise_std, observation_space, forecast_shape, increase_uncertainty=False):
        super().__init__(observation_space, forecast_shape)

        self.input_noise_std = noise_std
        self.increase_uncertainty = increase_uncertainty

        self._noise_size = self._forecast_shaped_space.shape
        self._noise_std = self._get_noise_std()

    def _get_noise_std(self):
        if self.increase_uncertainty:
            return self.input_noise_std * np.outer(
                1 + np.log(1 + np.arange(self._noise_size[0])),
                np.ones(self._noise_size[-1])
            )
        else:
            return self.input_noise_std

    def _get_noise(self, size):
        return np.random.normal(scale=self._noise_std, size=size)

    def _forecast(self, val_c, val_c_n, n):
        return val_c_n + self._get_noise(val_c_n.shape).reshape(val_c_n.shape)

    @property
    def noise_std(self):
        return self._noise_std

    @noise_std.setter
    def noise_std(self, value):
        pass

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
