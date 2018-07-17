import nifty5 as ift
import numpy as np


def plot_test():
    rg_space1 = ift.makeDomain(ift.RGSpace((100,)))
    rg_space2 = ift.makeDomain(ift.RGSpace((80, 80)))
    hp_space = ift.makeDomain(ift.HPSpace(64))
    gl_space = ift.makeDomain(ift.GLSpace(128))

    fft = ift.FFTOperator(rg_space2)

    field_rg1_1 = ift.Field.from_global_data(rg_space1, np.random.randn(100))
    field_rg1_2 = ift.Field.from_global_data(rg_space1, np.random.randn(100))
    field_rg2 = ift.Field.from_global_data(
        rg_space2, np.random.randn(80 ** 2).reshape((80, 80)))
    field_hp = ift.Field.from_global_data(hp_space, np.random.randn(12*64**2))
    field_gl = ift.Field.from_global_data(gl_space, np.random.randn(32640))
    field_ps = ift.power_analyze(fft.times(field_rg2))

    ## Start various plotting tests

    ift.plot(field_rg1_1, title='Single plot')
    ift.plot_finish()

    ift.plot(field_rg2, title='2d rg')
    ift.plot([field_rg1_1, field_rg1_2], title='list 1d rg', label=['1', '2'])
    ift.plot(field_rg1_2, title='1d rg, xmin, ymin', xmin=0.5, ymin=0.,
             xlabel='xmin=0.5', ylabel='ymin=0')
    ift.plot_finish(title='Three plots')

    ift.plot(field_hp, title='HP planck-color', colormap='Planck-like')
    ift.plot(field_rg1_2, title='1d rg')
    ift.plot(field_ps)
    ift.plot(field_gl, title='GL')
    ift.plot(field_rg2, title='2d rg')
    ift.plot_finish(nx=2, ny=3, title='Five plots')

if __name__ == '__main__':
    plot_test()
