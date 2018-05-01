import numpy as np
from .multi_linear_operator import MultiLinearOperator


class MultiEndomorphicOperator(MultiLinearOperator):
    """
    Class for multi endomorphic operators.

    By definition, domain and target are the same in
    EndomorphicOperator.
    """

    @property
    def target(self):
        """
        MultiDomain : returns :attr:`domain`

        Returns `self.domain`, because this is also the target domain
        for endomorphic operators.
        """
        return self.domain

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        """Generate a zero-mean sample

        Generates a sample from a Gaussian distribution with zero mean and
        covariance given by the operator. If from_inverse is True, the sample
        is drawn from the inverse of the operator.

        Returns
        -------
        MultiField
            A sample from the Gaussian of given covariance.
        """
        raise NotImplementedError
