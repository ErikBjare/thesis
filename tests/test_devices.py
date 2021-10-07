from time import sleep

import pytest

from typing import Generator
from eegwatch.devices import EEGDevice


@pytest.fixture(scope="session")
def device() -> Generator[EEGDevice, None, None]:
    """Test fixture for board"""
    device = EEGDevice.create(device_name="synthetic")
    with device:
        sleep(1)
        yield device


def test_check(device: EEGDevice):
    bads = device.check(max_uv_abs=300)
    print(bads)
    assert bads
    # Seems to blink between the two...
    # assert bads == ["F6", "F8"] or bads == ["F4", "F6", "F8"]
    # print(bads)
    # assert not bads


def test_get_data(device: EEGDevice):
    df = device.get_data(clear_buffer=False)
    print(df)
    assert not df.empty
