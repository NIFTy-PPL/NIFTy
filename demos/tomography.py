import nifty4 as ift
import numpy as np


if __name__ == "__main__":
    np.random.seed(13)

    s_space = ift.RGSpace([512, 512])
    h_space = s_space.get_default_codomain()
    ht = ift.HarmonicTransformOperator(h_space)

    pow_spec = (lambda k: 42 / (k+1)**4)

    S = ift.create_power_operator(h_space, power_spectrum=pow_spec)

    sh = S.draw_sample()
    ss = ht(sh)

    def make_random_los(n):
        starts = np.random.random((2, n))
        ends = np.random.random((2, n))
        return starts, ends

    nlos = 1000
    starts, ends = make_random_los(nlos)

    R = ift.library.LOSResponse(
        s_space, starts=starts, ends=ends,
        sigmas_up=np.full(nlos, 0.1), sigmas_low=np.full(nlos, 0.1))

    Rh = R*ht

    signal_to_noise = 10
    noise_amplitude = Rh(sh).val.std()/signal_to_noise
    N = ift.ScalingOperator(noise_amplitude**2, Rh.target)
    n = N.draw_sample()
    d = Rh(sh) + n
    j = Rh.adjoint_times(N.inverse_times(d))
    ctrl = ift.GradientNormController(name="Iter", tol_abs_gradnorm=1e-10,
                                      iteration_limit=300)
    inverter = ift.ConjugateGradient(controller=ctrl)
    Di = ift.library.WienerFilterCurvature(S=S, R=Rh, N=N, inverter=inverter)
    mh = Di.inverse_times(j)
    m = ht(mh)

    ift.plot(m, name="reconstruction.png")
    ift.plot(ss, name="signal.png")
    ift.plot(ht(j), name="j.png")
