import unittest
import numpy as np
from warnings import warn


class TestCase(unittest.TestCase):
    def assertEqual(self, first, second, msg=None) -> None:
        try:
            super().assertEqual(first, second, msg=msg)
        except (ValueError, AssertionError):
            try:
                np.testing.assert_equal(first, second, err_msg=msg if msg else '')
            except AssertionError as e:
                try:
                    np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-10, err_msg=msg if msg else '')
                except TypeError:
                    raise e
