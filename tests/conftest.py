import pytest


def _gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """The simulator is CuPy-only: skip every test when no CUDA device is available."""
    if _gpu_available():
        return
    skip = pytest.mark.skip(reason="CuPy with a CUDA device is required")
    for item in items:
        item.add_marker(skip)
