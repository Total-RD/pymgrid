import unittest
import numpy as np
from warnings import warn


class TestCase(unittest.TestCase):
    def assertEqual(self, first, second, msg=None) -> None:
        try:
            super().assertEqual(first, second, msg=msg)
        except ValueError:
            try:
                np.testing.assert_equal(first, second, err_msg=msg if msg else '')
            except AssertionError as e:
                np.testing.assert_allclose(first, second, err_msg=msg if msg else '')
                warn(f"Arrays are close but not equal: {e.args[0]}")
