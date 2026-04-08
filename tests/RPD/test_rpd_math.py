import unittest

import numpy as np

try:
    from .rpd_utils import rpd_matlab  # when run as a module
except Exception:  # pragma: no cover
    from tests.RPD.rpd_utils import rpd_matlab  # when run as a script


class TestRPDMath(unittest.TestCase):
    def test_rpd_flag1_matches_manual_columnwise(self):
        X = np.array([[10.0, 5.0], [12.0, 7.0], [11.0, 6.0]])
        got = rpd_matlab(X, flag=1)
        exp = np.array(
            [
                [(10 - 10) / 10, (5 - 5) / 5],
                [(12 - 10) / 10, (7 - 5) / 5],
                [(11 - 10) / 10, (6 - 5) / 5],
            ]
        )
        np.testing.assert_allclose(got, exp, rtol=0, atol=1e-12)

    def test_rpd_flag0_matches_manual_rowwise(self):
        X = np.array([[10.0, 5.0], [12.0, 7.0], [11.0, 6.0]])
        got = rpd_matlab(X, flag=0)
        exp = np.array(
            [
                [(10 - 5) / 5, (5 - 5) / 5],
                [(12 - 7) / 7, (7 - 7) / 7],
                [(11 - 6) / 6, (6 - 6) / 6],
            ]
        )
        np.testing.assert_allclose(got, exp, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

