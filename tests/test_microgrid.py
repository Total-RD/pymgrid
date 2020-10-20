"""
Copyright 2020 Total S.A.
Authors:Gonzague Henri <gonzague.henri@total.com>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/08/27 08:04 $
Gonzague Henri
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.pymgrid.MicrogridGenerator import MicrogridGenerator

import unittest

class TestMicrogrid(unittest.TestCase):

    def setUp(self):
        mgen = MicrogridGenerator()
        self.mg = mgen._create_microgrid()

    def test_set_horizon(self):
        self.mg.set_horizon(25)
        self.assertEqual(25, self.mg._horizon)


    def test_get_updated_values(self):
        mg_data = self.mg.get_updates_values()
        self.assertEqual(0, self.mg_data['pv'].iloc[0])

    def test_forecast_all(self):
        self.mg.set_horizon(24)
        forecast = self.mg.forecast_all()

        self.assertEqual(24, len(forecast['load']))


    def test_forecast_pv(self):
        self.mg.set_horizon(24)
        forecast = self.mg.forecast_pv()

        self.assertEqual (24, len(forecast['pv']))


    def test_forecast_load(self):
        self.mg.set_horizon(24)
        forecast = self.mg.forecast_load()

        self.assertEqual (24, len(forecast['load']))


    def test_run(self):
        pv1 = self.mg.forecast_pv()[1]
        control={}
        self.mg.run(control)
        pv2 =  self.mg.pv

        self.assertEqual(pv1, pv2)


    def test_train_test_split(self):
        self.mg.train_test_split()

        self.assertEqual('training',self.mg._data_set_to_use)

    def test_reset(self):
        control = {}
        self.mg.run(control)
        self.mg.reset()

        self.assertEqual (0, self.mg._tracking_timestep)




if __name__ == '__main__':
    unittest.main()
