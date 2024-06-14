import pytest
from scoringrules.visualization import reliability_diagram

from .conftest import OUT_DIR


@pytest.mark.parametrize("uncertainty_band", [None, "consistency", "confidence"])
def test_reliability_diagram(probability_forecasts, uncertainty_band):
    fct, obs = probability_forecasts.T

    ax = reliability_diagram(obs, fct, uncertainty_band=uncertainty_band)

    if OUT_DIR:
        fig = ax.get_figure()
        fig.savefig(OUT_DIR / f"reliability_diagram_{uncertainty_band}.png")
        fig.clf()
