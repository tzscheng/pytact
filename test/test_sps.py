import math
import unittest
import warnings

from tact.sim import _resolve_sim_dt


class SpsYamlTest(unittest.TestCase):
    def test_sps_is_converted_to_dt(self):
        self.assertEqual(_resolve_sim_dt({'sps': 240}, 0.001), 1.0 / 240.0)

    def test_dt_takes_precedence_and_warns(self):
        with self.assertWarnsRegex(UserWarning, "using 'dt'.*ignoring 'sps'"):
            dt = _resolve_sim_dt({'dt': 0.002, 'sps': 240}, 0.001)
        self.assertEqual(dt, 0.002)

    def test_invalid_sps_is_rejected(self):
        invalid_values = (0, -1, math.inf, "fast", None, True)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "sim.sps"):
                    _resolve_sim_dt({'sps': value}, 0.001)

    def test_dt_only_remains_unchanged(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dt = _resolve_sim_dt({'dt': 0.004}, 0.001)
        self.assertTrue(math.isclose(dt, 0.004))

    def test_absent_timestep_keeps_default(self):
        self.assertEqual(_resolve_sim_dt({}, 0.001), 0.001)


if __name__ == "__main__":
    unittest.main()
