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
from numpy.testing import assert_allclose


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pymgrid.MicrogridGenerator import MicrogridGenerator

import unittest


class TestMicogridGenerator(unittest.TestCase):

    def setUp(self):
        self.mgen = MicrogridGenerator()

    def test_get_random_file(self):
        import inspect, pymgrid
        from pathlib import Path

        path = Path(inspect.getfile(pymgrid)).parent
        path = path / 'data/pv'
        data = self.mgen._get_random_file(path)

        self.assertEqual(len(data), 8760)

    def test_scale_ts(self):
        ts = pd.DataFrame( [i for i in range(10)])
        factor = 4
        scaled = self.mgen._scale_ts(ts, factor)
        assert_allclose(ts/ts.sum()*factor, scaled)

    def test_get_genset(self):
        genset = self.mgen._get_genset()
        self.assertEqual (1000, genset['rated_power'])


    def test_get_battery(self):
        battery = self.mgen._get_battery()
        self.assertEqual (1000, battery['capa'])

    def test_get_grid_price_ts(self):
        price = self.mgen._get_grid_price_ts(10, price=0.2)
        self.assertTrue(all([p == 0.2 for p in price]))

    def test_get_grid(self):
        grid = self.mgen._get_grid()
        self.assertEqual(1000, grid['grid_power_import'])

    def test_size_mg(self):
        ts = pd.DataFrame([i for i in range(10)])
        mg = self.mgen._size_mg(ts, 10)

        self.assertEqual(18, mg['grid'])

    def test_size_genset(self):
        self.assertEqual(int(np.ceil(10/0.9)), self.mgen._size_genset([10, 10, 10]))

    def test_size_battery(self):
        size = self.mgen._size_battery([10, 10, 10])
        self.assertLessEqual(30, size)
        self.assertGreaterEqual(50, size)

    def test_generate_microgrid(self):
        microgrids = self.mgen.generate_microgrid().microgrids

        self.assertEqual(self.mgen.nb_microgrids, len(microgrids))

    def test_create_microgrid(self):
        mg = self.mgen._create_microgrid()

        self.assertEqual(1, mg.architecture['battery'])

if __name__ == '__main__':
    unittest.main()
