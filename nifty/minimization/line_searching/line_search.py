import abc

from keepers import Loggable

from nifty import LineEnergy


class LineSearch(object, Loggable):
    """
    Class for finding a step size.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):

        """
        Parameters
        ----------

        f : callable f(x, *args)
            Objective function.

        fprime : callable f'(x, *args)
            Objective functions gradient.

        f_args : tuple (optional)
            Additional arguments passed to objective function and its
            derivation.
        """

        self.pk = None
        self.line_energy = None
        self.f_k_minus_1 = None

    def _set_line_energy(self, energy, pk, f_k_minus_1=None):
        """
        Set the coordinates for a new line search.

        Parameters
        ----------
        xk : ndarray, d2o, field
            Starting point.

        pk : ndarray, d2o, field
            Unit vector in search direction.

        f_k : float (optional)
            Function value f(x_k).

        fprime_k : ndarray, d2o, field (optional)
            Function value fprime(xk).

        """
        self.line_energy = LineEnergy(position=0.,
                                      energy=energy,
                                      line_direction=pk)
        if f_k_minus_1 is not None:
            f_k_minus_1 = f_k_minus_1.copy()
        self.f_k_minus_1 = f_k_minus_1

    @abc.abstractmethod
    def perform_line_search(self, energy, pk, f_k_minus_1=None):
        raise NotImplementedError
