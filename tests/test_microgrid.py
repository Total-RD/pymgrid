import numpy as np
import pandas as pd
from pymgrid import MicrogridGenerator

mgen = MicrogridGenerator()
mg = mgen._create_microgrid()

def test_set_horizon():
    mg.set_horizon(25)
    assert(25, mg._horizon)


def test_get_updated_values():
    mg_data = mg.get_updates_values()
    assert(0, mg_data['pv'].iloc[0])

def test_forecast_all():
    mg.set_horizon(24)
    forecast = mg.forecast_all()

    assert(24, len(forecast['load']))


def test_forecast_pv():
    mg.set_horizon(24)
    forecast = mg.forecast_pv()

    assert (24, len(forecast['pv']))


def test_forecast_load():
    mg.set_horizon(24)
    forecast = mg.forecast_load()

    assert (24, len(forecast['load']))


def test_run():
    pv1 = mg.forecast_pv()[1]
    control={}
    mg.run(control)
    pv2 = mg.pv

    assert(pv1, pv2)


def test_train_test_split():
    mg.train_test_split()

    assert('training', mg._data_set_to_use)

def test_reset():
    control = {}
    mg.run(control)
    mg.reset()

    assert (0, mg._tracking_timestep)





