from copy import deepcopy

from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid

from pymgrid.algos import RuleBasedControl


class TestRBC(TestCase):
    def setUp(self) -> None:
        self.rbc = RuleBasedControl(get_modular_microgrid())

    def test_init(self):
        microgrid = get_modular_microgrid()
        self.assertEqual(microgrid, self.rbc.microgrid)
        self.assertEqual(microgrid, deepcopy(self.rbc).microgrid)

    def test_priority_list(self):
        rbc = deepcopy(self.rbc)

        for j, (element_1, element_2) in enumerate(zip(rbc.priority_list[:-1], rbc.priority_list[1:])):
            with self.subTest(testing=f'element_{j}<=element_{j+1}'):
                self.assertLessEqual(element_1.marginal_cost, element_2.marginal_cost)

    def test_run_once(self):
        rbc = deepcopy(self.rbc)

        self.assertEqual(len(rbc.microgrid.log), 0)

        n_steps = 10

        log = rbc.run(n_steps)

        self.assertEqual(len(log), n_steps)
        self.assertEqual(log, rbc.microgrid.log)
        return rbc

    def test_reset_after_run(self):
        rbc = self.test_run_once()
        rbc.reset()
        self.assertEqual(len(rbc.microgrid.log), 0)