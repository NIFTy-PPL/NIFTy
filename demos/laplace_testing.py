from nifty import *

import numpy as np
from nifty import Field,\
                  EndomorphicOperator,\
                  PowerSpace
import plotly.offline as pl
import plotly.graph_objs as go

import numpy as np
from nifty import Field, \
    EndomorphicOperator, \
    PowerSpace


class TestEnergy(Energy):
    def __init__(self, position, Op):
        super(TestEnergy, self).__init__(position)
        self.Op = Op

    def at(self, position):
        return self.__class__(position=position, Op=self.Op)

    @property
    def value(self):
        return 0.5 * self.position.dot(self.Op(self.position))

    @property
    def gradient(self):
        return self.Op(self.position)

    @property
    def curvature(self):
        curv = CurvOp(self.Op)
        return curv

class CurvOp(InvertibleOperatorMixin, EndomorphicOperator):
    def __init__(self, Op,inverter=None, preconditioner=None):
        self.Op = Op
        self._domain = Op.domain
        super(CurvOp, self).__init__(inverter=inverter,
                                                     preconditioner=preconditioner)

    def _times(self, x, spaces):
        return self.Op(x)

if __name__ == "__main__":

    distribution_strategy = 'not'

    # Set up position space
    s_space = RGSpace([128,128])
    # s_space = HPSpace(32)

    # Define harmonic transformation and associated harmonic space
    fft = FFTOperator(s_space)
    h_space = fft.target[0]

    # Setting up power space
    p_space = PowerSpace(h_space, logarithmic=False,
                         distribution_strategy=distribution_strategy, nbin=70)

    # Choosing the prior correlation structure and defining correlation operator
    pow_spec = (lambda k: (.05 / (k + 1) ** 2))
    # t = Field(p_space, val=pow_spec)
    t= Field.from_random("normal", domain=p_space)
    lap = LaplaceOperator(p_space)
    T = SmoothnessOperator(p_space,sigma=1.)
    test_energy = TestEnergy(t,T)

    def convergence_measure(a_energy, iteration): # returns current energy
        x = a_energy.value
        print (x, iteration)
    minimizer1 = VL_BFGS(convergence_tolerance=0,
                       iteration_limit=1000,
                       callback=convergence_measure,
                       max_history_length=3)


    def explicify(op, domain):
        space = domain
        d = space.dim
        res = np.zeros((d, d))
        for i in range(d):
            x = np.zeros(d)
            x[i] = 1.
            f = Field(space, val=x)
            res[:, i] = op(f).val
        return res
    A = explicify(lap.times, p_space)
    B = explicify(lap.adjoint_times, p_space)
    test_energy,convergence = minimizer1(test_energy)
    data = test_energy.position.val.get_full_data()
    pl.plot([go.Scatter(x=log(p_space.kindex)[1:], y=data[1:])], filename="t.html")
    tt = Field.from_random("normal", domain=t.domain)
    print "adjointness"
    print t.dot(lap(tt))
    print tt.dot(lap.adjoint_times(t))
    print "log kindex"
    aa = Field(p_space, val=p_space.kindex.copy())
    aa.val[0] = 1

    print lap(log(aa)).val
    print "######################"
    print test_energy.position.val