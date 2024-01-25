import jax
import numpy as np
import pytest

import nifty8.re as jft

pmp = pytest.mark.parametrize

def test_map_forest_out_axes():
    f = lambda x : x
    with pytest.raises(TypeError):
        # out_axes != 0
        jft.map_forest(
				f,
				out_axes = 1
		)

def test_map_forest_in_axes():
    f = lambda x : x
    jft.map_forest(
		f,
		in_axes = 1
    )
    with pytest.raises(ValueError):
        # len(in_axes) > 2
	    jft.map_forest(
	    	f,
	    	in_axes = (1,2)
	    )
    with pytest.raises(ValueError):
        # in_axes shoult be int or list of ints
	    jft.map_forest(
	    	f,
	    	in_axes = (None,)
	    )
    with pytest.raises(TypeError):
        # in_axes should not contain floats
	    jft.map_forest(
	    	f,
	    	in_axes = (1.,)
	    )
