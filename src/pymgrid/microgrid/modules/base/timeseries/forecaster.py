import numpy as np
from pandas.api.types import is_number


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
        _validate_callable_forecaster(forecaster, time_series)


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
                         f"\nFunc call forecaster(val_c={val_c}, val_c_plus_n={vector_true_forecast}, n={n})"
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
