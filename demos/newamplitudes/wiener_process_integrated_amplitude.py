import nifty5 as ift
import numpy as np


def mfplot(fld, A1, A2, name):
    p = ift.Plot()
    dom = fld.domain
    mi, ma = np.min(fld.val), np.max(fld.val)
    for ii in range(dom.shape[-1]):
        extr = ift.DomainTupleFieldInserter(op.target, 1, (ii,)).adjoint
        p.add(extr(fld), zmin=mi, zmax=ma)
    p.add(A1)
    p.add(A2)
    p.add(fld)
    p.output(name=name, xsize=14, ysize=14)


if __name__ == '__main__':
    np.random.seed(42)
    sspace = ift.RGSpace((512, 512))
    hspace = sspace.get_default_codomain()
    target = ift.PowerSpace(hspace,
                            ift.PowerSpace.useful_binbounds(hspace, True))
    # A = Amplitude(target, [5, -3, 1, 0], [1, 1, 1, 0.5],
    #               ['rest', 'smooth', 'wienersigma'])
    # fld = ift.from_random('normal', A.domain)
    # ift.extra.check_jacobian_consistency(A, fld)
    # op = CorrelatedFieldNormAmplitude(sspace, A, 3, 2)
    # plts = []
    # p = ift.Plot()
    # for _ in range(25):
    #     fld = ift.from_random('normal', op.domain)
    #     plts.append(A.force(fld))
    #     p.add(op(fld))
    # p.output(name='debug1.png', xsize=20, ysize=20)
    # ift.single_plot(plts, name='debug.png')

    # Multifrequency
    sspace1 = ift.RGSpace((512, 512))
    hspace1 = sspace1.get_default_codomain()
    target1 = ift.PowerSpace(hspace1,
                             ift.PowerSpace.useful_binbounds(hspace1, True))
    A1 = ift.WPAmplitude(target1, [0, -2, 0], [1E-5, 1, 1], 1, 0.99,
                         ['rest1', 'smooth1', 'wienersigma1'])
    sspace2 = ift.RGSpace((20,), distances=(1e7))
    hspace2 = sspace2.get_default_codomain()
    target2 = ift.PowerSpace(hspace2,
                             ift.PowerSpace.useful_binbounds(hspace2, True))
    A2 = ift.WPAmplitude(target2, [0, -2, 0], [1E-5, 1, 1], 1, 0.99,
                         ['rest2', 'smooth2', 'wienersigma2'])
    op = ift.CorrelatedFieldNormAmplitude((sspace1, sspace2), (A1, A2), 3, 2)
    for jj in range(10):
        fld = ift.from_random('normal', op.domain)
        mfplot(op(fld), A1.force(fld), A2.force(fld), 'debug{}.png'.format(jj))
