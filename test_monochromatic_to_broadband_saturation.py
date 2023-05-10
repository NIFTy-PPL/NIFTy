import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt

# set up test scenario
val_mf1 = np.zeros((256, 256, 256))
val_mf2 = np.zeros((256, 256, 256))
val_x = np.zeros((256, 256))

lambda_min = 400
lambda_max = 700

inp_log10_min = -3
inp_log10_max = 3

for i in range(256):
    # wavelengths
    val_x[i] = 1/((1/lambda_max) + i * ((1/lambda_min) - (1/lambda_max)) / (256 - 1))

    # field with constant intensity, single wavelength excitation in each row
    val_mf1[i, :, i] = 1.

    # field with decaying intensity along columns, single wavelength excitation in each row
    val_mf2[i, :, i] = np.geomspace(10.0 ** inp_log10_max, 10.0 ** inp_log10_min, 256)
    
# test mf conversion routines
try:
    # original mf conversion routines
    from nifty8.plot import _rgb_data
    val_mf1_rgb = _rgb_data(val_mf1)
    val_mf2_rgb = _rgb_data(val_mf2)
    suptitle = 'original'
except ImportError:
    # new mf conversion routines
    from nifty8.plot import SpectrumToRGBProjector
    mf_to_rgb_1 = SpectrumToRGBProjector(1e3)
    val_mf2 = np.log10(val_mf2.clip(1e-32, None)).clip(inp_log10_min, None) / 6 + 0.5
    print(np.min(val_mf2), np.max(val_mf2))

    if True:  # test luminance saturation
        mf_to_rgb_1 = SpectrumToRGBProjector(1e3, map_energies_logarithmically=False)
        lower = np.geomspace(1, 1e6, 257)
        mf_to_rgb_1.specify_input_spectrum_bins_via_bin_boundaries(lower[:-1], lower[1:])
        val_mf1_rgb = mf_to_rgb_1.project_integrated_flux(val_mf2)
        sat = 2.5
        mf_to_rgb_1.set_saturation_flux(sat)
        val_mf2_rgb = mf_to_rgb_1.project_integrated_flux(val_mf2)
        suptitle = 'log energy mapping, luminance saturation'
        str1 = 'no saturation point set'
        str2 = f'saturation point set to {sat:1.2f}'

    if False:  # test luminance saturation
        mf_to_rgb_1.specify_input_spectrum_bins_via_bin_boundaries(np.arange(1, 257), np.arange(2, 258))
        val_mf1_rgb = mf_to_rgb_1.project_integrated_flux(val_mf2)
        sat = 2.5
        mf_to_rgb_1.set_saturation_flux(sat)
        val_mf2_rgb = mf_to_rgb_1.project_integrated_flux(val_mf2)
        suptitle = 'luminance saturation'
        str1 = 'no saturation point set'
        str2 = f'saturation point set to {sat:1.2f}'
    if False:  # test cone response saturation
        mf_to_rgb_1.specify_input_spectrum_bins_via_bin_boundaries(np.arange(1, 257), np.arange(2, 258))
        val_mf1_rgb = mf_to_rgb_1.project_integrated_flux(val_mf2, saturation_via='retinal cone response')
        sat = 2.5
        mf_to_rgb_1.set_saturation_flux(2.5)
        val_mf2_rgb = mf_to_rgb_1.project_integrated_flux(val_mf2, saturation_via='retinal cone response')
        suptitle = 'cone response saturation'
        str1 = 'no saturation point set'
        str2 = f'saturation point set to {sat:1.2f}'

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 5))
fig.patch.set_facecolor('lightgrey')

extent = (lambda_max, lambda_min, inp_log10_min, inp_log10_max)
xticks = np.linspace(lambda_min, lambda_max, 5)
ytick_positions = np.linspace(inp_log10_min, inp_log10_max, inp_log10_max - inp_log10_min + 1)
ytick_labels = ['$10^{' + f'{v:1.0f}' + '}$' for v in ytick_positions]

ax1.imshow(val_mf1_rgb, extent=extent, aspect=50., origin='upper')
ax1.xaxis.set_ticks(xticks)
ax2.yaxis.set_ticks(ytick_positions, labels=ytick_labels)
ax1.set_xlabel('excitation wavelength')
ax1.set_ylabel('constant intensity')
ax1.set_title(str1)

plotval = val_mf2_rgb
print(plotval.max())
ax2.imshow(plotval, extent=extent, origin='upper', aspect=50.)
for i, color in enumerate(['tab:red', 'tab:green', 'tab:blue']):
    ax2.contour(plotval[..., i], levels=[0.9999,], extent=extent, colors='black', linewidths=0.2, origin='upper')
ax2.axhline(0.0, linestyle='--', color='black', linewidth=0.5, alpha=0.5)
ax2.xaxis.set_ticks(xticks)
ax2.yaxis.set_ticks(ytick_positions, labels=ytick_labels)
ax2.set_xlabel('excitation wavelength')
ax2.set_ylabel('intensity')
ax2.set_title(str2)

fig.suptitle(suptitle)
plt.tight_layout()
plt.savefig(suptitle.replace(' ', '_') + '.png', dpi=150)

# plot with nifty plotting routine
#dom_x = ift.RGSpace((256, 256))
#dom_f = ift.RGSpace(256)
#
#dom_mf = ift.DomainTuple.make((dom_x, dom_f))
#
#f_x = ift.makeField(dom_x, val_x)
#f_mf1 = ift.makeField(dom_mf, val_mf1)
#f_mf2 = ift.makeField(dom_mf, val_mf2)
#
#p = ift.Plot()
#p.add(f_x)
#p.add(f_mf1)
#p.add(f_mf2)
#p.output(nx=3, xsize=12, name=suptitle + '_niftyplot.png', dpi=150)
