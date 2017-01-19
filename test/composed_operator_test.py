# -*- coding: utf-8 -*-

from nifty import RGSpace, FFTOperator, ComposedOperator, Field, \
                  RGRGTransformation

x1 = RGSpace((8,))
x2 = RGSpace((6,))
y1 = RGRGTransformation.get_codomain(x1)
y2 = RGRGTransformation.get_codomain(x2)

fft1 = FFTOperator(x1)
fft2 = FFTOperator(x2)
ifft1 = FFTOperator(y1)
ifft2 = FFTOperator(y2)


com1 = ComposedOperator((fft1, fft2))
com2 = ComposedOperator((fft1, fft2, ifft1))

f = Field((x1, x2), val=0)
f.val[1,1] = 11

com1(f)
com2(f, spaces=(0,1,0))