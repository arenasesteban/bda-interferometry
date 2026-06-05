import json
from pathlib import Path

import pytest


@pytest.fixture
def real_antenna_config_path():
    path = Path("conf/antenna/alma.cfg")

    if not path.exists():
        pytest.skip(f"No se encontró archivo de antenas: {path}")

    return str(path)


@pytest.fixture
def real_simulation_config():
    path = Path("conf/runtime/simulation/alma-band-01.json")

    if not path.exists():
        pytest.skip(f"No se encontró archivo de simulación: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)