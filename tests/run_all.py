"""Run every check script in tests/ and print an overall summary.

Run from the repository root:  python tests/run_all.py   (or %run in Colab)
Figures are written to test_outputs/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import finish

import test_agar
import test_growth

if __name__ == "__main__":
    outcomes = {"agar": test_agar.main(), "growth": test_growth.main()}
    print("\n=== summary ===")
    for name, passed in outcomes.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    finish(all(outcomes.values()))
