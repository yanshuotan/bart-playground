import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests (slow, optional)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_bartz: marks tests that require bartz library"
    )
    config.addinivalue_line(
        "markers", "requires_pmlb: marks tests that require pmlb library"
    )

