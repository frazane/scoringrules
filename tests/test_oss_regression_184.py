import pytest

import scoringrules as sr


@pytest.mark.parametrize("score", [sr.rps_score, sr.rls_score])
@pytest.mark.parametrize("invalid_label", [-2, 0, 4, 1.5])
def test_ranked_scores_reject_invalid_category_labels(score, invalid_label, to_backend):
    fct = to_backend([[0.25, 0.5], [0.25, 0.25], [0.5, 0.25]])
    assert score(to_backend([1, 3]), fct, k_axis=0).shape == (2,)

    obs = to_backend([3, invalid_label])
    try:
        score(obs, fct, k_axis=0)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected, f"{score.__name__} accepted invalid category {invalid_label}"
