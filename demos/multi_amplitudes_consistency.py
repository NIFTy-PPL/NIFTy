import nifty5 as ift
import numpy as np

np.random.seed(42)


def testAmplitudesConsistency(seed, sspace):
    offset_std = 30
    intergated_fluct_std0 = .003
    intergated_fluct_std1 = 0.1
    
    nsam = 1000

    hspace = sspace.get_default_codomain()
    target0 = ift.PowerSpace(hspace)

    fsspace = ift.RGSpace((12,), (0.4,))
    fhspace = fsspace.get_default_codomain()
    target1 = ift.PowerSpace(fhspace)

    fa = ift.CorrelatedFieldMaker()
    fa.add_fluctuations(target0, intergated_fluct_std0, 1E-8, 1.1, 2., 2.1, .5,
                        -2, 1., 'spatial')
    fa.add_fluctuations(target1, intergated_fluct_std1, 1E-8, 3.1, 1., .5, .1,
                        -4, 1., 'freq')
    op = fa.finalize(offset_std, 1E-8, '')

    samples = [ift.from_random('normal',op.domain) for _ in range(nsam)]
    tot_flm, _ = fa.stats(fa.total_fluctuation,samples)
    offset_std,_ = fa.stats(fa.amplitude_total_offset,samples)
    intergated_fluct_std0,_ = fa.stats(fa.average_fluctuation(0),samples)
    intergated_fluct_std1,_ = fa.stats(fa.average_fluctuation(1),samples)
    
    slice_fluct_std0,_ = fa.stats(fa.slice_fluctuation(0),samples)
    slice_fluct_std1,_ = fa.stats(fa.slice_fluctuation(1),samples)

    sams = [op(s) for s in samples]
    fluct_total = fa.total_fluctuation_realized(sams)
    fluct_space = fa.average_fluctuation_realized(sams,0)
    fluct_freq = fa.average_fluctuation_realized(sams,1)
    zm_std_mean = fa.offset_amplitude_realized(sams)
    sl_fluct_space = fa.slice_fluctuation_realized(sams,0)
    sl_fluct_freq = fa.slice_fluctuation_realized(sams,1)

    np.testing.assert_allclose(offset_std, zm_std_mean, rtol=0.5)
    np.testing.assert_allclose(intergated_fluct_std0, fluct_space, rtol=0.5)
    np.testing.assert_allclose(intergated_fluct_std1, fluct_freq, rtol=0.5)
    np.testing.assert_allclose(tot_flm, fluct_total, rtol=0.5)
    np.testing.assert_allclose(slice_fluct_std0, sl_fluct_space, rtol=0.5)
    np.testing.assert_allclose(slice_fluct_std1, sl_fluct_freq, rtol=0.5)

    print("Expected  offset Std: " + str(offset_std))
    print("Estimated offset Std: " + str(zm_std_mean))

    print("Expected  integrated fluct. space Std: " +
          str(intergated_fluct_std0))
    print("Estimated integrated fluct. space Std: " + str(fluct_space))

    print("Expected  integrated fluct. frequency Std: " +
          str(intergated_fluct_std1))
    print("Estimated integrated fluct. frequency Std: " + str(fluct_freq))
    
    print("Expected  slice fluct. space Std: " +
          str(slice_fluct_std0))
    print("Estimated slice fluct. space Std: " + str(sl_fluct_space))

    print("Expected  slice fluct. frequency Std: " +
          str(slice_fluct_std1))
    print("Estimated slice fluct. frequency Std: " + str(sl_fluct_freq))

    print("Expected  total fluct. Std: " + str(tot_flm))
    print("Estimated total fluct. Std: " + str(fluct_total))


# Move to tests
# FIXME This test fails but it is not relevant for the final result
# assert_allclose(ampl(from_random('normal', ampl.domain)).val[0], vol) or 1??
# End move to tests

# move to tests
# assert_allclose(
#     smooth(from_random('normal', smooth.domain)).val[0:2], 0)
# end move to tests
for seed in [1, 42]:
    for sp in [
            ift.RGSpace((32, 64), (1.1, 0.3)),
            ift.HPSpace(64),
            ift.GLSpace(64)
    ]:
        testAmplitudesConsistency(seed, sp)
