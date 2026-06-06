from scoringrules.backend import backends


def test_default_active_is_numpy():
    # default active must be numpy so that an active value of "numba" can only
    # come from an explicit set_active call.
    assert backends._active == "numpy"
