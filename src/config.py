# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

_config = dict(
    hartley_convention="non_canonical_hartley",
)


def update(key, value, /):
    """Update the global configuration of NIFTy and NIFTy.re

    Parameters
    ----------
    key : str
        Identifier for the configuration option.
    value : Any
        Value for the configuration option.


    Currently, the following configuration options are available:

    - "hartley_convention": one of "non_canonical_hartley" or
      "canonical_hartley" for ducc's old non-canonical Hartley convention
      respectively ducc's new canononical Hartley convention
    """
    global _config
    if not isinstance(key, str):
        raise TypeError(f"key must be a string; got {key!r}")
    key = key.lower()
    if key == "hartley_convention":
        if not isinstance(value, str):
            raise TypeError(f"value to {key!r} must be a string; got {value!r}")
        if value in ("ducc_hartley", "non_canonical_hartley"):
            value = "non_canonical_hartley"
        elif value in ("ducc_fht", "canonical_hartley"):
            value = "canonical_hartley"
        else:
            raise ValueError(f"invalid value to {key!r}; got {value!r}")
    else:
        raise ValueError(f"invalid key; got {key!r}")
    _config[key] = value
