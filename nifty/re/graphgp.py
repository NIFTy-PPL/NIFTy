"""NIFTy.re adapter for the optional external :mod:`graphgp` package."""

from __future__ import annotations

from dataclasses import field
from functools import partial
from importlib import import_module
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .model import Initializer, Model
from .sharding import ShardingLayout
from .tree_math import ShapeWithDtype


def _require_graphgp():
    try:
        return import_module("graphgp")
    except ImportError as exc:
        raise ImportError(
            "GraphGPField requires the optional GraphGP dependency; install "
            "it with `pip install 'nifty[re_graphgp]'`."
        ) from exc


def _draw_sharded_normal(key, *, shape, dtype, sharding):
    draw = jax.jit(
        lambda k: jax.random.normal(k, shape=shape, dtype=dtype),
        out_shardings=sharding,
    )
    return draw(key)


class GraphGPField(Model):
    """A spatially sharded GraphGP correlated field.

    The external GraphGP package remains optional until this class is
    constructed. The returned array has the plan's logical global shape and
    is sharded over the mesh axis named ``"space"``.
    """

    plan: Any = field(metadata=dict(static=False))
    covariance: Any = field(metadata=dict(static=False))
    offset: Any = field(metadata=dict(static=False))
    mesh: Mesh
    prefix: str
    _name_exc: str
    _fixed_covariance: bool

    def __init__(
        self,
        plan,
        covariance,
        *,
        mesh,
        prefix="dust",
        offset=0.0,
    ):
        graphgp = _require_graphgp()
        if not isinstance(plan, graphgp.distributed.DistributedGraph):
            raise TypeError("plan must be a graphgp.distributed.DistributedGraph")
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a jax.sharding.Mesh")
        if "space" not in mesh.axis_names:
            raise ValueError("mesh must contain an axis named 'space'")
        if int(mesh.shape["space"]) != plan.n_partitions:
            raise ValueError("the mesh 'space' axis must match plan.n_partitions")

        fixed = isinstance(covariance, (tuple, list)) and len(covariance) == 2
        if fixed:
            covariance = tuple(jnp.asarray(v) for v in covariance)
        elif not isinstance(covariance, Model):
            raise TypeError(
                "covariance must be a (radii, values) pair or a NIFTy Model"
            )
        if not (isinstance(offset, Model) or isinstance(offset, float)):
            raise TypeError("offset must be a float or scalar NIFTy Model")

        self.plan = plan
        self.covariance = covariance
        self.offset = offset
        self.mesh = mesh
        self.prefix = str(prefix)
        self._name_exc = self.prefix + "excitations"
        self._fixed_covariance = fixed

        shape = tuple(plan.output_shape)
        dtype = jnp.result_type(plan.graph.points)
        axes = [None] * len(shape)
        axes[plan.output_axis] = "space"
        spec = P(*axes)
        sharding = NamedSharding(mesh, spec)
        domain = {self._name_exc: ShapeWithDtype(shape, dtype)}
        if not fixed:
            domain |= covariance.domain
        if isinstance(offset, Model):
            domain |= offset.domain

        init = Initializer(
            {
                self._name_exc: partial(
                    _draw_sharded_normal,
                    shape=shape,
                    dtype=dtype,
                    sharding=sharding,
                )
            }
        )
        if not fixed:
            init = init | covariance.init
        if isinstance(offset, Model):
            init = init | offset.init
        super().__init__(
            domain=domain,
            target=ShapeWithDtype(shape, dtype),
            init=init,
        )

    @property
    def excitation_name(self):
        return self._name_exc

    def position_specs(self):
        """Return this model's partial position-spec mapping."""
        axes = [None] * len(self.plan.output_shape)
        axes[self.plan.output_axis] = "space"
        return {self._name_exc: P(*axes)}

    def sharding_layout(self, *, sample_axis=None):
        """Return a layout that shards this field's excitation spatially."""
        return ShardingLayout(
            mesh=self.mesh,
            position_specs=self.position_specs(),
            sample_axis=sample_axis,
        )

    def __call__(self, x):
        graphgp = _require_graphgp()
        covariance = self.covariance if self._fixed_covariance else self.covariance(x)
        offset = self.offset(x) if isinstance(self.offset, Model) else self.offset
        field = graphgp.distributed.generate(
            self.plan,
            covariance,
            x[self._name_exc],
            mesh=self.mesh,
            axis_name="space",
        )
        return jnp.reshape(field, self.plan.output_shape) + offset
