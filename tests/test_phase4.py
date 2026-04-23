import json
import pytest
from pathlib import Path
import pandas as pd

import config


@pytest.mark.integration
def test_model_selection_json_exists():
    """MODEL-01: Verify model_selection.json exists."""
    path = Path(config.MODEL_SELECTION_PATH)
    assert path.exists(), "model_selection.json not written by Phase 4"


@pytest.mark.integration
def test_model_selection_schema():
    """MODEL-06: Verify JSON schema — all three approaches present."""
    path = Path(config.MODEL_SELECTION_PATH)
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    for approach in ["classical", "differencing", "harmonic_gls"]:
        assert approach in data, f"Missing approach: {approach}"
        assert "winner" in data[approach], f"Missing winner key in {approach}"
        assert "full_table" in data[approach], f"Missing full_table in {approach}"


@pytest.mark.integration
def test_acf_pacf_plots_exist():
    """MODEL-02: ACF/PACF plots generated per approach."""
    pass  # Wave 1: verify plot generation


@pytest.mark.integration
def test_causality_invertibility():
    """MODEL-03: Root moduli verification."""
    pass  # Wave 1: check root constraints


@pytest.mark.integration
def test_ljung_box_results():
    """MODEL-04: Ljung-Box test on top-3 models."""
    pass  # Wave 1: verify LB implementation


@pytest.mark.integration
def test_phase4_notebook_exists():
    """REPORT-01, REPORT-03: Phase4.ipynb exists."""
    path = Path("Phase4.ipynb")
    assert path.exists(), "Phase4.ipynb not created"
