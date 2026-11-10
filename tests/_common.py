"""Shared helpers for the runnable check scripts in tests/.

Every check script can be run directly, locally or in Colab:
    python tests/test_agar.py
    %run tests/test_agar.py          (Colab / Jupyter, from the repository root)
Figures are written to test_outputs/<script name>/ in the repository root.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def gpuAvailable():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class CheckRun:
    """Collects PASS/FAIL results and saves figures for one check script."""

    def __init__(self, name):
        self.name = name
        self.results = []
        self.outputDir = REPO_ROOT / "test_outputs" / name
        self.outputDir.mkdir(parents=True, exist_ok=True)
        print(f"=== {name} ===")

    def check(self, label, passed, detail=""):
        passed = bool(passed)
        self.results.append((label, passed))
        line = f"[{'PASS' if passed else 'FAIL'}] {label}"
        print(line + (f"  ({detail})" if detail else ""))
        return passed

    def saveFigure(self, figure, fileName):
        path = self.outputDir / fileName
        figure.savefig(path, dpi=120, bbox_inches="tight")
        print(f"       figure: {path.relative_to(REPO_ROOT)}")
        return path

    def summary(self):
        failed = [label for label, passed in self.results if not passed]
        print(f"--- {self.name}: {len(self.results) - len(failed)} passed, {len(failed)} failed")
        return not failed


def finish(passed):
    """Exit with a non-zero code on failure when run as a script (no-op on success)."""
    if not passed:
        sys.exit(1)
