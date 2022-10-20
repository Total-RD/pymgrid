from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.microgrid.envs import DiscreteMicrogridEnv


class TestDiscreteEnv(TestCase):
    def test_init(self):
        microgrid = get_modular_microgrid()
        env_2 = DiscreteMicrogridEnv(microgrid)

        print("Here")