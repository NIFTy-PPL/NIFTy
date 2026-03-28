import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional


def _get_n_dim_hann_window(shape, axes):
    """Generates an N-dimensional Hann window for specified axes."""
    window = jnp.ones(shape)
    for ax in axes:
        # Create 1D Hann window: 0.5 * (1 - cos(2 * pi * n / (N-1)))
        n = shape[ax]
        w_1d = jnp.hanning(n)

        # Reshape for broadcasting: e.g., (N,) -> (1, N, 1) for axis 1
        new_shape = [1] * len(shape)
        new_shape[ax] = n
        window *= w_1d.reshape(new_shape)
    return window


def compute_empirical_power_spectrum(
    field: jnp.ndarray,
    distances: Union[float, Tuple[float, ...]],
    axes: Optional[Union[int, Tuple[int, ...]]] = None,
    use_window: bool = False,
    n_bins: Optional[int] = 128,
):
    """Computes the average empirical power spectrum of a field across the given
       axes, assuming regular grids.

    Parameters
    ----------
    field: jnp.array
        Field of which to compute the power spectrum
    distances: float, Tuple[float]
        Pixel distances of `field` along the axes to be converted
    axes: int, Tuple[int], None
        Axes over which to compute the power spectrum. If None, all axes are
        consumed
    use_window: bool
        Whether to apply a Hann window to the field along the consumed axes
        before computing the power. Required if the field does not have periodic
        boundaries.
    n_bins: Optional[int]
        Maximum number of k bins to use

    Returns
    -------
    ps: jnp.array
        Empirical power spectrum
    k_bin_centers: jnp.array
        Geometric means of k bins
    """
    if axes is None:
        axes = tuple(range(field.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    ndim_included = len(axes)

    # Validate distances
    if isinstance(distances, (int, float)):
        dist_tuple = (float(distances),) * ndim_included
    else:
        dist_tuple = distances
        if len(dist_tuple) != ndim_included:
            raise ValueError(
                f"Length of distances ({len(dist_tuple)}) must match "
                f"number of included axes ({ndim_included})."
            )

    if use_window:
        window = _get_n_dim_hann_window(field.shape, axes)
        field = field * window

        # Correct for power loss: Windowing reduces the total variance.
        # We must divide by the 'window power factor' (mean of squared window)
        w_correction = jnp.mean(window**2)
    else:
        w_correction = 1.0

    # Compute raw power of fourier modes
    power_raw = (jnp.abs(jnp.fft.fftn(field, axes=axes)) ** 2) / w_correction

    # Generate logarithmic k bins
    axis_lengths = jnp.array(
        [field.shape[ax] * dist_tuple[i] for i, ax in enumerate(axes)]
    )

    k_min = 1.0 / jnp.max(axis_lengths)
    k_max = 1.0 / jnp.min(jnp.array(dist_tuple))

    k_bins = jnp.geomspace(k_min, k_max, n_bins + 1)
    k_bin_centers = jnp.sqrt(k_bins[1:] * k_bins[:-1])

    # Assign Fourier components to k bins
    ks = [
        jnp.fft.fftfreq(field.shape[ax], d=dist_tuple[i]) for i, ax in enumerate(axes)
    ]
    k_grids = jnp.meshgrid(*ks, indexing="ij")
    k_mag = jnp.sqrt(sum(k**2 for k in k_grids)).flatten()
    indices = jnp.digitize(k_mag, k_bins) - 1
    bin_counts = jax.ops.segment_sum(jnp.ones_like(k_mag), indices, num_segments=n_bins)

    entries_present = bin_counts > 0

    k_bin_centers = k_bin_centers[entries_present]
    bin_counts = bin_counts[entries_present]

    # Reorder and reshape field dimensions to simplify vmapping over the unconsumed axes
    all_axes = list(range(field.ndim))
    other_axes = [a for a in all_axes if a not in axes]
    n_pix_in_consumed_axes = jnp.prod(jnp.array([field.shape[ax] for ax in axes]))

    power_raw = jnp.transpose(power_raw, other_axes + list(axes))
    power_raw = power_raw.reshape(-1, n_pix_in_consumed_axes)

    # Compute average power in bins
    def average_power_per_k_bin(power_vals):
        bin_sum = jax.ops.segment_sum(
            power_vals.flatten(), indices, num_segments=n_bins
        )
        return bin_sum[entries_present] / bin_counts

    ps = jax.vmap(average_power_per_k_bin)(power_raw)

    # Reshape to restore unconsumed axes
    final_shape = [field.shape[ax] for ax in other_axes] + [len(k_bin_centers)]
    ps = ps.reshape(final_shape)

    return ps, k_bin_centers
