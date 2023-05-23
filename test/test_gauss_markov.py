import numpy as np
import nifty8 as ift


def check_linear_wpprior():
    sp1 = ift.RGSpace(10, distances=1)
    sp2 = ift.RGSpace([3, 3])
    dom = ift.makeDomain([sp2, sp1])

    amp = ift.Field.from_raw(dom, 1)
    wp = ift.WPPrior(amp, space=1)
    xi = ift.from_random(wp.domain)
    res = wp(xi)

    x1 = res.val
    x2 = xi["xi"].val

    diff = np.zeros(x1.shape)
    diff[:, :, 1:] = np.diff(x1, axis=2)
    diff[:, :, 0] = x1[:, :, 0]
    np.testing.assert_allclose(diff, x2)

    # do linear operator checks
    ift.extra.check_linear_operator(wp)


def check_non_linear_wpprior():
    sp1 = ift.RGSpace(10, distances=1)
    sp2 = ift.RGSpace([3, 3])
    dom = ift.makeDomain([sp2, sp1])

    amp = ift.ScalingOperator(dom, 1).ducktape("amplitude")
    wp = ift.WPPrior(amp, space=1)

    one_field = ift.MultiField.full(amp.domain, 1)
    xi = ift.from_random(wp.domain)
    xi_amp_one = ift.MultiField.union((xi, one_field))
    res = wp(xi_amp_one)

    x1 = res.val
    x2 = xi["xi"].val
    diff = np.zeros(x1.shape)
    diff[:, :, 0] = x1[:, :, 0]
    diff[:, :, 1:] = np.diff(x1, axis=2)
    np.testing.assert_allclose(diff, x2)

    # do the operator checks for non-linear ops
    ift.extra.check_operator(wp, xi)


check_linear_wpprior()
check_non_linear_wpprior()

# plt.plot(x1, label="cumsum", alpha=0.3)
# plt.plot(x2, label="xi")
# plt.plot(diff, label="diff", alpha=0.3)
# plt.legend()
# plt.show()
