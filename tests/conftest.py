import pytest
import os

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


def _setup_ray_local():
    """Setup Ray in local mode, removing RAY_ADDRESS if set."""
    import ray
    original_ray_address = os.environ.pop('RAY_ADDRESS', None)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=2, _node_ip_address='127.0.0.1', include_dashboard=False)
    return original_ray_address


def _teardown_ray_local(original_ray_address):
    """Cleanup Ray and restore RAY_ADDRESS if it existed."""
    import ray
    ray.shutdown()
    if original_ray_address is not None:
        os.environ['RAY_ADDRESS'] = original_ray_address


@pytest.fixture(scope="session", autouse=True)
def auto_ray_local():
    """Ensure Ray runs in local mode for all tests, preventing cluster connection timeouts."""
    original_ray_address = _setup_ray_local()
    yield
    _teardown_ray_local(original_ray_address)

