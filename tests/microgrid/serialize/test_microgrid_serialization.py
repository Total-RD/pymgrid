import numpy as np

from pymgrid import Microgrid

from tests.helpers.modular_microgrid import get_modular_microgrid
from tests.helpers.test_case import TestCase


class TestMicrogridSerialization(TestCase):
    def test_serialize_no_modules(self):
        microgrid = Microgrid([], add_unbalanced_module=False)
        dump = microgrid.dump()
        loaded = Microgrid.load(dump)

        self.assertEqual(microgrid, loaded)

    def test_serialize_with_renewable(self):
        microgrid = get_modular_microgrid(remove_modules=["genset", "battery", "load", "grid"],
                                          add_unbalanced_module=False)

        self.assertEqual(len(microgrid.modules), 1)
        self.assertEqual(microgrid, Microgrid.load(microgrid.dump()))
