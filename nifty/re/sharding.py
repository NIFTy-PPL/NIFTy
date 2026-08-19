"""Sharding descriptions for spatial and sample-parallel inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .tree_math import Vector


P = PartitionSpec


def _lookup_spec(specs, key, default=P()):
    if specs is None:
        return default
    if isinstance(specs, Vector):
        specs = specs.tree
    if isinstance(specs, dict):
        return specs.get(key, default)
    return specs


def _spec_tree(value, specs):
    """Expand a partial specification to the structure of ``value``."""
    if isinstance(value, Vector):
        inner = specs.tree if isinstance(specs, Vector) else specs
        return Vector(_spec_tree(value.tree, inner))
    if isinstance(value, dict):
        return {
            key: _spec_tree(val, _lookup_spec(specs, key))
            for key, val in value.items()
        }
    if isinstance(value, tuple):
        if isinstance(specs, (tuple, list)) and not isinstance(specs, P):
            return tuple(_spec_tree(v, specs[i]) for i, v in enumerate(value))
        return tuple(_spec_tree(v, specs) for v in value)
    if isinstance(value, list):
        if isinstance(specs, (tuple, list)) and not isinstance(specs, P):
            return [_spec_tree(v, specs[i]) for i, v in enumerate(value)]
        return [_spec_tree(v, specs) for v in value]
    return specs if isinstance(specs, P) else P()


def _prepend_axis(spec, axis):
    spec = spec if isinstance(spec, P) else P()
    return P(axis, *tuple(spec))


@dataclass(eq=False, frozen=True)
class ShardingLayout:
    """Describe position and sample shardings on a named JAX mesh.

    ``position_specs`` may be a partial mapping. Leaves not mentioned in it
    are replicated. Sample arrays gain ``sample_axis`` as their leading
    sharded dimension while retaining their position sharding afterwards.
    """

    mesh: Mesh
    position_specs: Any = None
    sample_axis: str | None = "sample"

    __hash__ = object.__hash__

    def __post_init__(self):
        if not isinstance(self.mesh, Mesh):
            raise TypeError("mesh must be a jax.sharding.Mesh")
        if self.sample_axis is not None and self.sample_axis not in self.mesh.axis_names:
            raise ValueError(
                f"sample axis {self.sample_axis!r} is not present in mesh axes "
                f"{self.mesh.axis_names!r}"
            )

    @property
    def replicated(self):
        return NamedSharding(self.mesh, P())

    @property
    def key_sharding(self):
        if self.sample_axis is None:
            return self.replicated
        return NamedSharding(self.mesh, P(self.sample_axis))

    @property
    def sample_axis_size(self):
        if self.sample_axis is None:
            return 1
        return int(self.mesh.shape[self.sample_axis])

    def position_specs_for(self, value):
        return _spec_tree(value, self.position_specs)

    def position_shardings(self, value):
        specs = self.position_specs_for(value)
        return _map_specs(specs, lambda spec: NamedSharding(self.mesh, spec))

    def sample_specs_for(self, value):
        if self.sample_axis is None:
            raise ValueError("sample_axis is required for sharding sample arrays")
        return _map_specs(
            self.position_specs_for(value),
            lambda spec: _prepend_axis(spec, self.sample_axis),
        )

    def sample_shardings(self, value):
        specs = self.sample_specs_for(value)
        return _map_specs(specs, lambda spec: NamedSharding(self.mesh, spec))


def _map_specs(tree, fn):
    if isinstance(tree, Vector):
        return Vector(_map_specs(tree.tree, fn))
    if isinstance(tree, dict):
        return {key: _map_specs(val, fn) for key, val in tree.items()}
    if isinstance(tree, tuple) and not isinstance(tree, P):
        return tuple(_map_specs(val, fn) for val in tree)
    if isinstance(tree, list):
        return [_map_specs(val, fn) for val in tree]
    return fn(tree)
