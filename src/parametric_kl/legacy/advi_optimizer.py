import  nifty5 as ift

class ADVIOptimizer():

    def __init__(self, H, model, entropy,initial_position ,eta=1, alpha=1, tau=1, epsilon=1e-16):
        self.model = model
        self.H = H
        self.entropy = entropy
        self.alpha = alpha
        self.eta = eta
        self.tau=tau
        self.epsilon = epsilon

        self.counter = 1

        E = ift.ParametrizedGaussianKL(initial_position, H, self.model,
                                       self.entropy, 5)
        self.s = E.gradient**2


    def _step(self, position, gradient):
        self.s = self.alpha * gradient**2 + (1-self.alpha)*self.s
        self.rho = self.eta * self.counter**(-0.5+ self.epsilon) / (self.tau + ift.sqrt(self.s))
        new_position = position - self.rho * gradient
        self.counter += 1
        return new_position

    def __call__(self, x,steps,N_samples=1, N_validate=10):
        for i in range(steps):

            E = ift.ParametrizedGaussianKL(x, self.H, self.model,
                                           self.entropy, N_samples)
            x = self._step(E.position, E.gradient)

        return x

class MGVIOptimizer():

    def __init__(self, H, initial_position ,eta=1, alpha=1, tau=1, epsilon=1e-16):
        self.H = H
        self.alpha = alpha
        self.eta = eta
        self.tau=tau
        self.epsilon = epsilon

        self.counter = 1

        E = ift.MetricGaussianKL(initial_position, H, 5,
                                   mirror_samples=False)
        self.s = E.gradient**2


    def _step(self, position, gradient):
        self.s = self.alpha * gradient**2 + (1-self.alpha)*self.s
        self.rho = self.eta * self.counter**(-0.5+ self.epsilon) / (self.tau + ift.sqrt(self.s))
        new_position = position - self.rho * gradient
        self.counter += 1
        return new_position

    def __call__(self, x,steps,N_samples=1, N_validate=10, mirror_samples=False):
        for i in range(steps):

            E = ift.MetricGaussianKL(x, self.H, N_samples,
                                     mirror_samples=mirror_samples)
            x = self._step(E.position, E.gradient)

        return x