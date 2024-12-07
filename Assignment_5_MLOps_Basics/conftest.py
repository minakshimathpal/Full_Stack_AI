import pytest
import os

@pytest.fixture
def models_dir():
    """Provide the directory where models are saved."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "models")