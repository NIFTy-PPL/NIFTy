
import unittest

from numpy.testing import assert_equal, assert_almost_equal

from nifty import *

from itertools import product
from test.common import expand
from test.common import generate_spaces

np.random.seed(42)


class ConjugateGradient_Tests(unittest.TestCase):
    spaces = generate_spaces()


    @expand(product(spaces, [10,  100, 1000], [1E-3, 1E-4, 1E-5], [2, 3, 4] ))
    def test_property(self, space, iteration_limit, convergence_tolerance, 
                      convergence_level):
        
        x0 = Field.from_random('normal', domain=space)
        A = DiagonalOperator(space, diagonal = 1.)
        b = Field(space, val=0.)
        
        minimizer = ConjugateGradient(iteration_limit=iteration_limit,
                                    convergence_tolerance=convergence_tolerance, 
                                    convergence_level=convergence_level)
                                    
        (position, convergence) = minimizer(A=A, x0=x0, b=b)
        
        if position.domain[0] != space:
            raise TypeError
        if type(convergence) != int:
            raise TypeError

    @expand(product(spaces, [10,  100, 1000], [1E-3, 1E-4, 1E-5], [2, 3, 4] ))
    def test_property(self, space, iteration_limit, convergence_tolerance, 
                      convergence_level):
        
        x0 = Field.from_random('normal', domain=space)
        test_x = Field(space, val = 1.)
        A = DiagonalOperator(space, diagonal = 1.)
        b = Field(space, val=1.)
        
        minimizer = ConjugateGradient(iteration_limit=iteration_limit,
                                    convergence_tolerance=convergence_tolerance, 
                                    convergence_level=convergence_level)
                                    
        (position, convergence) = minimizer(A=A, x0=x0, b=b)
        
        assert_almost_equal(position.val.get_full_data(), 
                            test_x.val.get_full_data(), decimal=3)
        assert_equal(convergence, convergence_level+1)


