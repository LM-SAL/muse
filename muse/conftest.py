import pytest

from muse.tests.helpers import fake_response, fake_vdem, fake_vdem_offgrid


@pytest.fixture(scope="session")
def response():
    return fake_response()


@pytest.fixture(scope="session")
def vdem():
    return fake_vdem()


@pytest.fixture(scope="session")
def vdem_offgrid():
    return fake_vdem_offgrid()
