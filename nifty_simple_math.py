## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.


##-----------------------------------------------------------------------------
import numpy as np
#from nifty.nifty_core import field
from nifty_about import about


def _math_helper(x, function):
    try:
        return x.apply_scalar_function(function)
    except(AttributeError):
        return function(np.array(x))

def cos(x):
    """
        Returns the cos of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        cosx : {scalar, array, field}
            Cosine of `x` to the specified base.

        See Also
        --------
        sin
        tan

        Examples
        --------
        >>> cos([-1,1])
        array([ 0.54030231,  0.54030231])
        >>> cos(field(point_space(2), val=[10, 100])).val
        array([ 0.54030231,  0.54030231])
    """
    return _math_helper(x, np.cos)
    
def sin(x):
    """
        Returns the sine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        sinx : {scalar, array, field}
            Sine of `x` to the specified base.

        See Also
        --------
        cos
        tan

        Examples
        --------
        >>> sin([-1,1])
        array([-0.84147098,  0.84147098])
        >>> sin(field(point_space(2), val=[-1, 1])).val
        array([-0.84147098,  0.84147098])

    """
    return _math_helper(x, np.sin)
    
def cosh(x):
    """
        Returns the hyperbolic cosine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        coshx : {scalar, array, field}
            cosh of `x` to the specified base.

        See Also
        --------
        sinh
        tanh

        Examples
        --------
        >>> cosh([-1,1])
        array([ 1.54308063,  1.54308063])
        >>> cosh(field(point_space(2), val=[-1, 1])).val
        array([ 1.54308063,  1.54308063])

    """
    return _math_helper(x, np.cosh)

def sinh(x):
    """
        Returns the hyperbolic sine  of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        sinhx : {scalar, array, field}
            sinh of `x` to the specified base.

        See Also
        --------
        cosh
        tanh

        Examples
        --------
        >>> sinh([-1,1])
        array([-1.17520119,  1.17520119])
        >>> sinh(field(point_space(2), val=[-1, 1])).val
        array([-1.17520119,  1.17520119])

    """
    return _math_helper(x, np.sinh)

def tan(x):
    """
        Returns the tangent of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        tanx : {scalar, array, field}
            Tangent of `x` to the specified base.

        See Also
        --------
        cos
        sin

        Examples
        --------
        >>> tan([10,100])
        array([ 0.64836083, -0.58721392])
        >>> tan(field(point_space(2), val=[10, 100])).val
        array([ 0.64836083, -0.58721392])

    """
    return _math_helper(x, np.tan)

def tanh(x):
    """
        Returns the hyperbolic tangent of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        tanhx : {scalar, array, field}
            tanh of `x` to the specified base.

        See Also
        --------
        cosh
        sinh

        Examples
        --------
        >>> tanh([-1,1])
        array([-0.76159416,  0.76159416])
        >>> tanh(field(point_space(2), val=[-1, 1])).val
        array([-0.76159416,  0.76159416])
    """
    return _math_helper(x, np.tanh)


def arccos(x):
    """
        Returns the arccosine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arccosx : {scalar, array, field}
            arccos of `x` to the specified base.

        See Also
        --------
        arcsin
        arctan

        Examples
        --------
        >>> arccos([-1,1])
        array([ 3.14159265,  0.        ])
        >>> arccos(field(point_space(2), val=[-1, 1])).val
        array([ 3.14159265,  0.        ])

    """
    return _math_helper(x, np.arccos)


def arcsin(x):
    """
        Returns the arcsine of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arcsinx : {scalar, array, field}
            Logarithm of `x` to the specified base.

        See Also
        --------
        arccos
        arctan

        Examples
        --------
        >>> arcsin([-1,1])
        array([-1.57079633,  1.57079633])
        >>> arcsin(field(point_space(2), val=[-1, 1])).val
        array([-1.57079633,  1.57079633])

    """
    return _math_helper(x, np.arcsin)


def arccosh(x):
    """
        Returns the hyperbolic arccos of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arccoshx : {scalar, array, field}
            arccos of `x` to the specified base.

        See Also
        --------
        arcsinh
        arctanh

        Examples
        --------
        >>> arcosh([1,10])
        array([ 0.        ,  2.99322285])
        >>> arccosh(field(point_space(2), val=[1, 10])).val
        array([ 0.        ,  2.99322285])
    """
    return _math_helper(x, np.arccosh)


def arcsinh(x):
    """
        Returns the hypberbolic sin of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arcsinhx : {scalar, array, field}
            arcsinh of `x` to the specified base.

        See Also
        --------
        arccosh
        arctanh

        Examples
        --------
        >>> arcsinh([1,10])
        array([ 0.88137359,  2.99822295])
        >>> arcsinh(field(point_space(2), val=[1, 10])).val
        array([ 0.88137359,  2.99822295])
    """
    return _math_helper(x, np.arcsinh)

def arctan(x):
    """
        Returns the arctan of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arctanx : {scalar, array, field}
            arctan of `x` to the specified base.

        See Also
        --------
        arccos
        arcsin

        Examples
        --------
        >>> arctan([1,10])
        array([ 0.78539816,  1.47112767])
        >>> arctan(field(point_space(2), val=[1, 10])).val
        array([ 0.78539816,  1.47112767])
    """
    return _math_helper(x, np.arctan)

def arctanh(x):
    """
        Returns the hyperbolic arc tangent of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        arctanhx : {scalar, array, field}
            arctanh of `x` to the specified base.

        See Also
        --------
        arccosh
        arcsinh

        Examples
        --------
        >>> arctanh([0,0.5])
        array([ 0.        ,  0.54930614])
        >>> arctanh(field(point_space(2), val=[0, 0.5])).val
        array([ 0.        ,  0.54930614])
    """
    return _math_helper(x, np.arctanh)

def sqrt(x):
    """
        Returns the square root of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        sqrtx : {scalar, array, field}
            Square root of `x`.

        Examples
        --------
        >>> sqrt([10,100])
        array([ 10.       ,  31.6227766])
        >>> sqrt(field(point_space(2), val=[10, 100])).val
        array([ 10.       ,  31.6227766])

    """
    return _math_helper(x, np.sqrt)

def exp(x):
    """
        Returns the exponential of a given object.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.

        Returns
        -------
        expx : {scalar, array, field}
            Exponential of `x` to the specified base.

        See Also
        --------
        log

        Examples
        --------
        >>> exp([10,100])
        array([  2.20264658e+04,   2.68811714e+43])
        >>> exp(field(point_space(2), val=[10, 100])).val
        array([  2.20264658e+04,   2.68811714e+43])

    """
    return _math_helper(x, np.exp)

def log(x,base=None):
    """
        Returns the logarithm with respect to a specified base.

        Parameters
        ----------
        x : {scalar, list, array, field}
            Input argument.
        base : {scalar, list, array, field}, *optional*
            Base of the logarithm (default: Euler's number).

        Returns
        -------
        logx : {scalar, array, field}
            Logarithm of `x` to the specified base.

        See Also
        --------
        exp

        Examples
        --------
        >>> log([100, 1000], base=10)
        array([ 2.,  3.])
        >>> log(field(point_space(2), val=[100, 1000]), base=10).val
        array([ 2.,  3.])

    """
    if(base is None):
        return _math_helper(x, np.log)

    base = np.array(base)
    if(np.all(base>0)):
        return _math_helper(x, np.log)/np.log(base)
    else:
        raise ValueError(about._errors.cstring("ERROR: invalid input basis."))

def conjugate(x):
    """
        Computes the complex conjugate of a given object.

        Parameters
        ----------
        x : {ndarray, field}
            The object to be complex conjugated.

        Returns
        -------
        conjx : {ndarray,field}
            The complex conjugated object.
    """        
    return _math_helper(x, np.conjugate)

def direct_dot(x, y):
    ## the input could be fields. Try to extract the data
    try:
        x = x.get_val()
    except(AttributeError):
        pass
    ## try to make a direct vdot
    try:
        return x.vdot(y)
    except(AttributeError):
        pass
    
    try:
        return y.vdot(x)
    except(AttributeError):
        pass        

    ## fallback to numpy 
    return np.vdot(x, y)    
    
    
        
        
        
        

##---------------------------------