"""
Copyright 2020 Total S.A.
Authors:Gonzague Henri <gonzague.henri@total.com>
Permission to use, modify, and distribute this software is given under the
terms of the pymgrid License.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2020/08/27 08:04 $
Gonzague Henri
"""
import unittest
import numpy as np

from pymgrid.MicrogridGenerator import MicrogridGenerator


class TestNonmodularMicrogrid(unittest.TestCase):
    def setUp(self):
        mgen = MicrogridGenerator()
        self.mg = mgen._create_microgrid()

    @staticmethod
    def random_control():
        return dict(pv_consummed=np.random.rand(),
                    battery_charge=np.random.rand(),
                    battery_discharge=np.random.rand(),
                    grid_import=np.random.rand(),
                    grid_export=np.random.rand()
                    )

    def test_set_horizon(self):
        self.mg.set_horizon(25)
        self.assertEqual(25, self.mg.horizon)

    def test_get_updated_values(self):
        mg_data = self.mg.get_updated_values()
        self.assertEqual(0, mg_data['pv'])

    def test_forecast_all(self):
        self.mg.set_horizon(24)
        forecast = self.mg.forecast_all()
        self.assertEqual(24, len(forecast['load']))

    def test_forecast_pv(self):
        self.mg.set_horizon(24)
        forecast = self.mg.forecast_pv()
        self.assertEqual (24, len(forecast))

    def test_forecast_load(self):
        self.mg.set_horizon(24)
        forecast = self.mg.forecast_load()
        self.assertEqual (24, len(forecast))

    def test_run(self):
        pv1 = self.mg.forecast_pv()[1]
        self.mg.run(self.random_control())
        pv2 = self.mg.pv
        self.assertEqual(pv1, pv2)

    def test_train_test_split(self):
        self.mg.train_test_split()
        self.assertEqual('training',self.mg._data_set_to_use)

    def test_reset(self):
        self.mg.run(self.random_control())
        self.mg.reset()
        self.assertEqual (0, self.mg._tracking_timestep)


if __name__ == '__main__':
    unittest.main()
