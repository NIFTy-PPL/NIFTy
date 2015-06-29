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

## TODO: optimize
## TODO: doc strings

from __future__ import division
import numpy as np
#import gfft as gf


def smooth_power(power, k, exclude=1, smooth_length=None):
    """
    """
    if smooth_length == 0:
        # No smoothing requested, just return the input array.
        return power

    if (exclude > 0):
        k = k[exclude:]
        excluded_power = np.copy(power[:exclude])
        power = power[exclude:]

    kmin = k[0]
    kdiff = np.abs(k[:-1] - k[1:])

    # Use the maximum difference between k values to define the bin width.
    # This should ensure that we always have some points in each bin.
    dk = np.max(kdiff)
    nk = int(max(k - kmin) / dk + 1.)

    if (smooth_length is None) or (smooth_length < 0):
        #smooth_length = np.abs(k[0]-k[-1])/(3*nk)
        smooth_length = dk

    # Store the number of data points that have contributed to each bin
#    counter = np.zeros(nk, dtype=float)
    counter = np.zeros(nk, dtype=np.int)
    # The binned power spectrum
    pbinned = np.zeros(nk, dtype=power.dtype)
    # do the binning
    for i in range(len(k)):
        ndx = int((k[i] - kmin) / dk)
        pbinned[ndx] += power[i]
        counter[ndx] += 1
    pbinned = pbinned / counter

    nmirror = int(5 * smooth_length / dk) + 2
    zpbinned = np.r_[np.exp(2 * np.log(pbinned[0]) - np.log(pbinned[1:nmirror][::-1])), pbinned,
                     np.exp(2 * np.log(pbinned[-1])- np.log(pbinned[-nmirror:-1][::-1]))]
    zpbinned = np.maximum(0, zpbinned)

    tpbinned = np.fft.fftshift(np.fft.fft(zpbinned))
    tkernel = gaussian_kernel(zpbinned.size, dk, smooth_length)
    pbinned2 = np.fft.ifft(np.fft.ifftshift(tpbinned * tkernel))

    pbinned2 = np.abs(pbinned2[nmirror - 1:-nmirror + 1])

    # Un-bin the results
    power2 = np.zeros(len(power), dtype=power.dtype)
    for i in range(len(k)):
        ndx = int((k[i] - kmin) / dk)
        power2[i] = pbinned2[ndx]

    if(exclude > 0):
        return np.r_[excluded_power, power2]
    else:
        return power2


def smooth_power_bf(power, k, exclude=1, smooth_length=None):

    if smooth_length == 0:
        # No smoothing requested, just return the input array.
        return power

    if (exclude > 0):
        k = k[exclude:]
        excluded_power = np.copy(power[:exclude])
        power = power[exclude:]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = k[1]-k[0]

    nmirror = int(5*smooth_length/(k[1]-k[0]))+2
    mpower = np.r_[np.exp(2*np.log(power[0])-np.log(power[1:nmirror][::-1])),power,np.exp(2*np.log(power[-1])-np.log(power[-nmirror:-1][::-1]))]
    mk = np.r_[(2*k[0]-k[1:nmirror][::-1]),k,(2*k[-1]-k[-nmirror:-1][::-1])]
    mdk = np.r_[0.5*(mk[1]-mk[0]),0.5*(mk[2:]-mk[:-2]),0.5*(mk[-1]-mk[-2])]

    p_smooth = np.empty(mpower.shape)
    for i in xrange(len(p_smooth)):
        C = np.exp(-(mk-mk[i])**2/(2.*smooth_length**2))*mdk
        p_smooth[i] = np.sum(C*mpower)/np.sum(C)

    p_smooth = p_smooth[nmirror - 1:-nmirror + 1]

#    dk = 0.5*(k[2:] - k[:-2])
#    dk = np.r_[0.5*(k[1]-k[0]),dk]
#    dk = np.r_[dk,0.5*(k[-1]-k[-2])]
#    if (smooth_length is None) or (smooth_length < 0):
#        smooth_length = k[1]-k[0]
#
#    p_smooth = np.empty(power.shape)
#    for i in xrange(len(p_smooth)):
#        C = np.exp(-(k-k[i])**2/(2.*smooth_length**2))*dk
#        p_smooth[i] = np.sum(C*power)/np.sum(C)

    if (exclude > 0):
        p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth


def smooth_power_2s(power, k, exclude=1, smooth_length=None):

    if smooth_length == 0:
        # No smoothing requested, just return the input array.
        return power

    if (exclude > 0):
        k = k[exclude:]
        excluded_power = np.copy(power[:exclude])
        power = power[exclude:]

    if (smooth_length is None) or (smooth_length < 0):
        smooth_length = k[1]-k[0]

    nmirror = int(5*smooth_length/(k[1]-k[0]))+2
    mpower = np.r_[np.exp(2*np.log(power[0])-np.log(power[1:nmirror][::-1])),power,np.exp(2*np.log(power[-1])-np.log(power[-nmirror:-1][::-1]))]
    mk = np.r_[(2*k[0]-k[1:nmirror][::-1]),k,(2*k[-1]-k[-nmirror:-1][::-1])]
    mdk = np.r_[0.5*(mk[1]-mk[0]),0.5*(mk[2:]-mk[:-2]),0.5*(mk[-1]-mk[-2])]

    p_smooth = np.empty(mpower.shape)
    for i in xrange(len(p_smooth)):
        l = i-int(2*smooth_length/mdk[i])-1
        l = max(l,0)
        u = i+int(2*smooth_length/mdk[i])+2
        u = min(u,len(p_smooth))
        C = np.exp(-(mk[l:u]-mk[i])**2/(2.*smooth_length**2))*mdk[l:u]
        p_smooth[i] = np.sum(C*mpower[l:u])/np.sum(C)

    p_smooth = p_smooth[nmirror - 1:-nmirror + 1]

#    dk = 0.5*(k[2:] - k[:-2])
#    dk = np.r_[0.5*(k[1]-k[0]),dk]
#    dk = np.r_[dk,0.5*(k[-1]-k[-2])]
#    if (smooth_length is None) or (smooth_length < 0):
#        smooth_length = k[1]-k[0]
#
#    p_smooth = np.empty(power.shape)
#    for i in xrange(len(p_smooth)):
#        l = i-int(2*smooth_length/dk[i])-1
#        l = max(l,0)
#        u = i+int(2*smooth_length/dk[i])+2
#        u = min(u,len(p_smooth))
#        C = np.exp(-(k[l:u]-k[i])**2/(2.*smooth_length**2))*dk[l:u]
#        p_smooth[i] = np.sum(C*power[l:u])/np.sum(C)

    if (exclude > 0):
        p_smooth = np.r_[excluded_power,p_smooth]
    return p_smooth


def smooth_field(val, fourier, zero_center, enforce_hermitian_symmetry, vol, \
    smooth_length=0.):
        """
        A function to smooth a regularly spaced field with a Gaussian smoothing
        kernel.

        Args:
            val: An ndarray containing the field values to be smoothed.
            fourier: A boolean indicating whether the field is in Fourier space
                or not.
            zero_center: A list of booleans stating whether the field is
                zero centered or not along each axis.
            enforce_hermitian_symmetry: A flag indicating whether the input
                field should be considered one half of a hermetian symmetric field
            vol: An iterable containing the pixel space volume of each axis on
                the regular grid.
            smooth_length: The standard deviation of the Gaussian smoothing
                kernel, in units of dist. [0]

        Returns:
            sval: The smoothed field values.
        """

#        if smooth_length == 0.:
#            return val
#
        if(fourier):
            tfield = val
            vol = 1/np.array(val.shape)/vol
        else:

            if(zero_center):
                tfield = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(val)))
            else:
                tfield = np.fft.fftshift(np.fft.fftn(val))
#            # Transform back to the signal space using GFFT.
#            tfield = gf.gfft(val, ftmachine='fft', \
#                in_zero_center=zero_center, out_zero_center=True, \
#                enforce_hermitian_symmetry=enforce_hermitian_symmetry, W=-1, \
#                alpha=-1, verbose=False)

        # Construct the Fourier transformed smoothing kernel
        tkernel = gaussian_kernel(val.shape, vol, smooth_length)
        # Multiply the smoothing kernel and the transformed spectrum
        tfield = tfield*tkernel
#
        if(fourier):
            sfield = tfield
        else:

            if(zero_center):
                sfield = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(tfield)))
            else:
                sfield = np.fft.ifftn(np.fft.ifftshift(tfield))
#            # Transform back to the signal space using GFFT.
#            sfield = gf.gfft(tfield, ftmachine='ifft', \
#                in_zero_center=True, out_zero_center=zero_center, \
#                enforce_hermitian_symmetry=enforce_hermitian_symmetry, W=-1, \
#                alpha=-1, verbose=False)

        return sfield


def gaussian_kernel(nx, dx, smooth_length):
    """
    Returns an image of the Fourier transform of a Gaussian smoothing kernel.

    Args:
        nx: A scalar or iterable of axis lengths defining the grid on which the
            smoothing kernel shall be applied.
        dx: A scalar or iterable of pixel volumes defining the grid on which the
            smoothing kernel shall be applied.
        smooth_length: The standard deviation of the Gaussian smoothing kernel,
            defined in units of dx.

    Returns:
        kernel: The Fourier transform of the Gaussian smoothing kernel.
    """
    ## FIXME: out source lambda function

    # The Fourier transform of a Gaussian function having amplitude 1.
    func = lambda x: np.exp(-2.*np.pi**2*x**2*smooth_length**2)

    ## TODO: check necessity of checks

    if np.isscalar(nx) and np.isscalar(dx):
        ndim = 1
        nx = [nx]
        dx = [dx]
    elif np.isscalar(nx) and not np.isscalar(dx) or \
        not np.isscalar(nx) and np.isscalar(dx):
            raise Exception("nx and dx must both be scalars or arrays. "+\
                "No mixtures allowed!")
    else:
        ndim = len(nx)

        if len(dx) != ndim:
            raise Exception("nx and dx must have the same length!")

    k = []
    shp = ()

    for i in range(ndim):
        dk = 1./nx[i]/dx[i]
        nk = nx[i]

        k.append(-0.5*nk*dk + np.arange(nk)*dk)
        print k
        shp=shp+(nk,)

    # The image of the Fourier transformed kernel.
    kernel = np.zeros(shp, dtype=float)
    kernel = kernel.flatten()

    for i in range(len(kernel)):
        ndx = np.unravel_index(i, shp)
        val = 1.
        # We assume the Gaussian profile is symmetric if in higher dimensions
        for j in range(ndim):
            val = val*func(k[j][ndx[j]])
        kernel[i] = val

    kernel = kernel.reshape(shp)

    return kernel