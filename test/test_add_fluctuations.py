import jax
import numpy as np
import pytest

import nifty8.re as jft

pmp = pytest.mark.parametrize


@pmp("flu", ([1e-1], [1e-1, 5e-3, 5e-3], np.array([1e-1, 5e-3]), 1e-1))
@pmp("slp", ([1e-1], [1e-1, 5e-3, 5e-3],  np.array([-1., 1e-2]), 1e-1))
@pmp("flx", ([1e-1], [1e-1, 5e-3, 5e-3], np.array([1e-1, 5e-3]), 1e-1))
@pmp("asp", ([1e-1], [1e-1, 5e-3, 5e-3], np.array([1e-1, 5e-3]), 1e-1))

def test_callable_add_fluctuation_inputs(flu, slp, flx, asp):
    with pytest.raises(TypeError):
        cf_zm = dict(offset_mean=0., offset_std=(1e-3, 1e-4))
        cf_fl = dict(
            fluctuations=flu,
            loglogavgslope=slp,
            flexibility=flx,
            asperity=asp,
        )
        cfm = jft.CorrelatedFieldMaker("cf")
        cfm.set_amplitude_total_offset(**cf_zm)
        cfm.add_fluctuations(
            shape=(16,),
            distances=tuple(1. / d for d in (16,)),
            **cf_fl,
            prefix="ax1",
            non_parametric_kind="power"
        )
        cfm.finalize()