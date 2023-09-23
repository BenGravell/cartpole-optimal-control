import constants


def test_dimensional_consistency():
    """Dimensional consistency checks"""
    assert len(constants.STATE_FIELDS) == constants.DIM_STATE
    assert len(constants.ACTION_FIELDS) == constants.DIM_ACTION
