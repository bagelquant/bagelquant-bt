"""A read-only client must not load the solver or statistical runtime."""

import subprocess
import sys


def test_public_import_defers_scipy_until_numerical_work() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bagelquant_bt as bt; "
            "assert callable(bt.evaluate_alpha); "
            "assert callable(bt.evaluate_weights); "
            "assert callable(bt.evaluate_execution); "
            "assert callable(bt.partial_rank_ic); "
            "assert callable(bt.hac_mean_test); "
            "assert 'scipy.optimize' not in sys.modules; "
            "assert 'scipy.stats' not in sys.modules; "
            "assert 'cvxpy' not in sys.modules; "
            "assert 'clarabel' not in sys.modules",
        ],
        check=True,
    )
