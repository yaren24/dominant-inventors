"""
Shared setup for the tests.

Two jobs: make `src` importable when pytest runs from the top of the repository,
and give every test an easy way to build a config that differs from the real one
in exactly one place.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.lib import config as config_module  # noqa: E402  (needs the path set first)

DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@pytest.fixture
def base_values():
    """The real config as a plain dictionary, for tests to modify and reload."""
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def make_config(tmp_path):
    """
    Build a config from a dictionary and load it properly, so that validation runs.

        cfg = make_config(values)                       # expect it to load
        with pytest.raises(config_module.ConfigError):  # expect it to be refused
            make_config(broken_values)
    """
    counter = {"n": 0}

    def build(values):
        counter["n"] += 1
        path = tmp_path / f"config_{counter['n']}.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        return config_module.load(path)

    return build


@pytest.fixture
def logging_config(base_values, make_config, tmp_path):
    """A config whose logs and outputs go to a temporary folder, for funnel tests."""
    base_values["paths"]["logs"] = str(tmp_path / "runs")
    base_values["paths"]["outputs"] = str(tmp_path / "outputs")
    return make_config(base_values)


@pytest.fixture
def dominance_config(base_values, make_config, tmp_path):
    """
    A config identical to the real one except that every path is temporary.

    A FACTORY rather than a config, because the counting method is the argument:
    tests that drive step 03 need the same tables built both ways. Item 8 needed
    that for the sum-to-one invariant, which holds under both counting methods
    where the per-patent one does not; item 9 needs it to compare a derived
    full count against a real full build.

    Written for tests/test_shares_sum_to_one.py and moved here when
    tests/test_task2b_item9.py became the second file to want it — a fixture in a
    test module is not visible to another test module, and conftest is pytest's
    own answer to that.
    """
    def build(counting_method="fractional"):
        base_values["counting"]["method"] = counting_method
        base_values["paths"]["interim"] = str(tmp_path / "interim")
        base_values["paths"]["processed"] = str(tmp_path / "processed")
        base_values["paths"]["logs"] = str(tmp_path / "runs")
        base_values["paths"]["outputs"] = str(tmp_path / "outputs")
        return make_config(base_values)

    return build
