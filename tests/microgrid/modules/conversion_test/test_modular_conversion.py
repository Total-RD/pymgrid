import numpy as np

from tests.helpers.test_case import TestCase


class TestToModular(TestCase):
    def setUp(self) -> None:
        from pymgrid.MicrogridGenerator import MicrogridGenerator
        mgen = MicrogridGenerator()
        mgen.generate_microgrid(modular=False)
        self.weak_grids = [microgrid for microgrid in mgen.microgrids if self.is_weak_grid(microgrid)]
        self.genset_only = [microgrid for microgrid in mgen.microgrids if not microgrid.architecture["grid"]]
        self.strong_grid_only = [microgrid for microgrid in mgen.microgrids if
                                 (not microgrid.architecture["genset"]) and self.is_strong_grid(microgrid)]
        self.strong_grid_and_genset = [microgrid for microgrid in mgen.microgrids if
                                       microgrid.architecture["genset"] and self.is_strong_grid(microgrid)]


    @staticmethod
    def is_weak_grid(microgrid):
        return microgrid.architecture["grid"] and microgrid._grid_status_ts.min().item() < 1

    @staticmethod
    def is_strong_grid(microgrid):
        return microgrid.architecture["grid"] and microgrid._grid_status_ts.min().item() == 1

    def test_weak_grid_conversion_success(self):
        for microgrid in self.weak_grids:
            modular_microgrid = microgrid.to_modular()
            self.assertTrue(modular_microgrid.grid.item().weak_grid)

    def test_genset_only(self):
        for microgrid in self.genset_only:
            modular = microgrid.to_modular()
            self.assertTrue(len(modular.genset) == 1)

            genset_module = modular.genset[0]
            self.assertEqual(microgrid.genset.fuel_cost, genset_module.genset_cost)
            self.assertEqual(microgrid.genset.co2, genset_module.co2_per_unit)
            self.assertEqual(microgrid.genset.rated_power*microgrid.genset.p_max, genset_module.max_production)

            with self.assertRaises(AttributeError):
                _ = modular.grid