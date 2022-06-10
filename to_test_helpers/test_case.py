import unittest
import numpy as np


class TestCase(unittest.TestCase):
    def assertEqual(self, first, second, msg=None) -> None:
        try:
            return super().assertEqual(first, second, msg=msg)
        except ValueError:
            if msg is None:
                return np.testing.assert_equal(first, second)
            else:
                return np.testing.assert_equal(first, second, err_msg=msg)

