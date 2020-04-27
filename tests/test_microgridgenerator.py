import numpy as np
import pandas as pd
from pymgrid import MicrogridGenerator
import os
import sys

mgen = MicrogridGenerator()

def test_get_random_file():
    path = os.path.split(os.path.dirname(sys.modules['pymgrid'].__file__))[0]
    path = path+'/data/pv/'
    mgen._get_random_file(path)
    assert(pd.columns,'Data')

def test_scale_ts():
    ts =pd.DataFrame( [i for i in range(10)])
    assert(ts.sum()*4,mgen._scale_ts(ts, 4).sum() )

def test_resize_timeseries():
    np.test()
    ts = pd.DataFrame([i for i in range(10)])
    assert (ts.shape[0] * 4, mgen._resize_timeseries(ts,1, 0.25).shape[0])


def test_get_genset():
    genset = mgen._get_genset()
    assert (1000, genset['rated_power'])


def test_get_battery():
    battery = mgen._get_battery()
    assert (1000, battery['capacity'])

def test_get_grid_price_ts():
    price = mgen._get_grid_price_ts(0.2, 10)
    assert (0.2, price[8])

def test_get_grid():
    grid = mgen.MicrogridGenerator()
    assert(1000, grid['grid_power_import'])

def test_generate_weak_grid_profile():
    outage = mgen._generate_weak_grid_profile(1,24,10)
    assert(0, outage.iloc[0,0])

def test_size_mg():
    ts = pd.DataFrame([i for i in range(10)])
    mg = mgen._size_mg(ts, 10)

    assert(20, mg['grid'])

def test_size_genset():
    assert (10/0.9, mgen._size_genset([10, 10, 10]))

def test_size_battery():

    assert(40, mgen._size_battery([10, 10, 10]))

def test_generate_microgrid():
    microgrids = mgen.generate_microgrid()

    assert (mgen.nb_microgrids, len(microgrids))

def test_create_microgrid():

    mg = mgen._create_microgrid()

    assert(1, mg.architecture['battery'])


