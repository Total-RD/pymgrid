"""
Copyright 2020 Total S.A.
Authors:Gonzague Henri <gonzague.henri@total.com>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/08/27 08:04 $
Gonzague Henri
"""

import numpy as np
import pandas as pd


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.pymgrid import MicrogridGenerator

import unittest


class TesttMicogridGenerator(unittest.TestCase):

    def setUp(self):
        self.mgen = MicrogridGenerator()

    def test_get_random_file(self):
        path = os.path.split(os.path.dirname(sys.modules['pymgrid'].__file__))[0]
        path = path+'/data/pv/'
        self.mgen._get_random_file(path)
        self.assertEqual(pd.columns,'Data')

    def test_scale_ts(self):
        ts =pd.DataFrame( [i for i in range(10)])
        self.assertEqual(ts.sum()*4,mgen._scale_ts(ts, 4).sum() )

    def test_resize_timeseries(self):
        np.test()
        ts = pd.DataFrame([i for i in range(10)])
        self.assertEqual (ts.shape[0] * 4, self.mgen._resize_timeseries(ts,1, 0.25).shape[0])


    def test_get_genset(self):
        genset = self.mgen._get_genset()
        self.assertEqual (1000, genset['rated_power'])


    def test_get_battery(self):
        battery = self.mgen._get_battery()
        self.assertEqual (1000, battery['capacity'])

    def test_get_grid_price_ts(self):
        price = self.mgen._get_grid_price_ts(0.2, 10)
        self.assertEqual (0.2, price[8])

    def test_get_grid(self):
        grid = self.mgen.MicrogridGenerator()
        self.assertEqual(1000, grid['grid_power_import'])

    def test_generate_weak_grid_profile(self):
        outage = self.mgen._generate_weak_grid_profile(1,24,10)
        self.assertEqual(0, outage.iloc[0,0])

    def test_size_mg(self):
        ts = pd.DataFrame([i for i in range(10)])
        mg = self.mgen._size_mg(ts, 10)

        self.assertEqual(20, mg['grid'])

    def test_size_genset(self):
        self.assertEqual (10/0.9, self.mgen._size_genset([10, 10, 10]))

    def test_size_battery(self):
        self.assertEqual(40, self.mgen._size_battery([10, 10, 10]))

    def test_generate_microgrid(self):
        microgrids = self.mgen.generate_microgrid()

        self.assertEqual (self.mgen.nb_microgrids, len(microgrids))

    def test_create_microgrid(self):
        mg = self.mgen._create_microgrid()

        self.assertEqual(1, mg.architecture['battery'])

if __name__ == '__main__':
    unittest.main()
