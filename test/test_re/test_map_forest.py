import jax
import numpy as np
import pytest

import nifty8.re as jft

pmp = pytest.mark.parametrize

def test_map_forest():
    f = lambda x : x
    with pytest.raises(TypeError):
        jft.map_forest(
				f,
				out_axes = 1
		)

def test_map_forest():
    f = lambda x : x
    jft.map_forest(
		f,
		in_axes = 1
    )
    with pytest.raises(ValueError):
	    jft.map_forest(
	    	f,
	    	in_axes = (1,2)
	    )
    with pytest.raises(ValueError):
	    jft.map_forest(
	    	f,
	    	in_axes = (None,)
	    )
    with pytest.raises(TypeError):
	    jft.map_forest(
	    	f,
	    	in_axes = (1.,)
	    )
