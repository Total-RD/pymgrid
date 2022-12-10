from tests.helpers.modular_microgrid import get_modular_microgrid
from tests.helpers.test_case import TestCase


class TestContainer(TestCase):
    def test_container_init(self):
        microgrid = get_modular_microgrid()
        self.assertTrue(len(microgrid.controllable.sources))
        self.assertTrue(len(microgrid.controllable.sinks))
        action = microgrid.sample_action()