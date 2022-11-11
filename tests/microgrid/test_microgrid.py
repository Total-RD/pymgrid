import numpy as np

from pymgrid import Microgrid

from tests.helpers.modular_microgrid import get_modular_microgrid
from tests.helpers.test_case import TestCase


class TestMicrogrid(TestCase):
    def test_from_scenario(self):
        for j in range(25):
            with self.subTest(microgrid_number=j):
                microgrid = Microgrid.from_scenario(j)
                self.assertTrue(hasattr(microgrid, "load"))
                self.assertTrue(hasattr(microgrid, "pv"))
                self.assertTrue(hasattr(microgrid, "battery"))
                self.assertTrue(hasattr(microgrid, "grid") or hasattr(microgrid, "genset"))
