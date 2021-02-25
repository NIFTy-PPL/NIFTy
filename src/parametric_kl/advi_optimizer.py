from parametric_KL import ParametricGaussianKL
import nifty7 as ift
import numpy as np

class ADVIOptimizer(ift.Minimizer):

    def __init__(self, steps,eta=1, alpha=1, tau=1, epsilon=1e-16):
        self.alpha = alpha
        self.eta = eta
        self.tau=tau
        self.epsilon = epsilon
        self.counter = 1
        self.steps = steps
        self.s = None

    def _step(self, position, gradient):
        self.s = self.alpha * gradient**2 + (1-self.alpha)*self.s
        self.rho = self.eta * self.counter**(-0.5+ self.epsilon) / (self.tau + ift.sqrt(self.s))
        new_position = position - self.rho * gradient
        self.counter += 1
        return new_position

    def __call__(self, E):
        if self.s is None:
            self.s = E.gradient**2
        convergence = 0
        for i in range(self.steps):
            x = self._step(E.position, E.gradient)
            # maybe some KL function for resample?
            E = ParametricGaussianKL.make(x, E._hamiltonian, E._variational_model,
                                E._n_samples, E._mirror_samples)

        return E, convergence

    def reset(self):
        self.counter = 0
        self.s = None