import pytest

from muse.tests.helpers import fake_response, fake_vdem, fake_vdem_offgrid
from muse.transforms.transforms import reshape_x_to_slit_step


@pytest.fixture(scope="session")
def response():
    return fake_response()


# Function-scoped on purpose: tests del attrs, chunk, and deep-copy the result.
@pytest.fixture
def raster(vdem):
    # nslits=35, nraster=11 are the defaults, so the recorded HISTORY string is unchanged.
    return reshape_x_to_slit_step(vdem)


@pytest.fixture(scope="session")
def vdem():
    return fake_vdem()


@pytest.fixture(scope="session")
def vdem_offgrid():
    return fake_vdem_offgrid()
