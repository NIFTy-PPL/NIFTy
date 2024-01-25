import pytest

import nifty8.re as jft


def test_map_forest_axes_validation():
    f = lambda x: x
    jft.map_forest(f, in_axes=1)
    with pytest.raises(ValueError):
        jft.map_forest(f, in_axes=(None, ))
    with pytest.raises(TypeError):
        jft.map_forest(f, in_axes=(1., ))
