# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

_config = dict(
    hartley_convention="non_canonical_hartley",
)


def update(key, value, /):
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
