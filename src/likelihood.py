from typing import Union, Optional

from jax import jvp, vjp
from jax import numpy as np
from jax.tree_util import Partial, tree_leaves, all_leaves, tree_map

from .optimize import cg
from .sugar import is1d, random_like, random_like_shapewdtype, sum_of_squares


class ShapeWithDtype():
    """Minimal helper class storing the shape and dtype of an object.

    Notes
    -----
    This class may not be transparent to JAX as it shall not be flattened
    itself. If used in a tree-like structure. It should only be used as leave.
    """
    def __init__(self, shape: Union[tuple, list], dtype=None):
        """Instantiates a storage unit for shape and dtype.

        Parameters
        ----------
        shape : tuple or list of int
            One-dimensional sequence of integers denoting the length of the
            object along each of the object's axis.
        dtype : dtype
            Data-type of the to-be-described object.
        """
        if not is1d(shape):
            ve = f"invalid shape; got {shape!r}"
            return ValueError(ve)

        self._shape = shape
        self._dtype = np.float64 if dtype is None else dtype

    @classmethod
    def from_leave(cls, element):
        """Convenience method for creating an instance of `ShapeWithDtype` from
        an object.

        To map a whole tree-like structure to a its shape and dtype use JAX's
        `tree_map` method like so:

            tree_map(ShapeWithDtype.from_leave, tree)

        Parameters
        ----------
        element : tree-like structure
            Object from which to take the shape and data-type.

        Returns
        -------
        swd : instance of ShapeWithDtype
            Instance storing the shape and data-type of `element`.
        """
        import numpy as onp

        if not all_leaves((element, )):
            ve = "tree is not flat and still contains leaves"
            raise ValueError(ve)
        return cls(np.shape(element), onp.common_type(element))

    @property
    def shape(self):
        """Retrieves the shape."""
        return self._shape

    @property
    def dtype(self):
        """Retrieves the data-type."""
        return self._dtype

    def __repr__(self):
        nm = self.__class__.__name__
        return f"{nm}(shape={self.shape}, dtype={self.dtype})"


class Likelihood():
    """Storage class for keeping track of the energy, the associated
    left-square-root of the metric and the metric.
    """
    def __init__(
        self,
        energy: callable,
        left_sqrt_metric: Optional[callable] = None,
        metric: Optional[callable] = None,
        lsm_tangents_shape=None
    ):
        """Instantiates a new likelihood.

        Parameters
        ----------
        energy : callable
            Function evaluating the negative log-likelihood.
        left_sqrt_metric : callable, optional
            Function applying the left-square-root of the metric.
        metric : callable, optional
            Function applying the metric.
        lsm_tangents_shape : tree-like structure of ShapeWithDtype, optional
            Structure of the data space.
        """
        self._hamiltonian = energy
        self._left_sqrt_metric = left_sqrt_metric
        self._metric = metric

        if lsm_tangents_shape is not None:
            if is1d(lsm_tangents_shape):
                lsm_tangents_shape = ShapeWithDtype(lsm_tangents_shape)
            else:
                leaves = tree_leaves(lsm_tangents_shape)
                if not all(isinstance(e, ShapeWithDtype) for e in leaves):
                    te = "objects of invalid type in tangent shapes"
                    raise TypeError(te)
        self._lsm_tan_shp = lsm_tangents_shape

    def __call__(self, primals):
        """Convenience method to access the `energy` method of this instance.
        """
        return self._hamiltonian(primals)

    def energy(self, primals):
        """Applies the metric at `primals` to `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the energy.

        Returns
        -------
        energy : float
            Energy at the position `primals`.
        """
        return self._hamiltonian(primals)

    def metric(self, primals, tangents):
        """Applies the metric at `primals` to `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric.

        Returns
        -------
        naturally_curved : tree-like structure
            Tree-like structure of the same type as primals to which the metric
            has been applied to.
        """
        if self._metric is None:
            # `left_sqrt_metric` is linear at any given position and thus the
            # position at which the derivative of this linear operator is taken
            # does not matter
            lsm_at_p = Partial(self.left_sqrt_metric, primals)
            arbitrary_lsm_tan_pos = tree_map(
                lambda x: np.ones(x.shape, dtype=x.dtype),
                self.left_sqrt_metric_tangents_shape
            )
            _, rsm_at_p = vjp(lsm_at_p, arbitrary_lsm_tan_pos)
            res = lsm_at_p(rsm_at_p(tangents)[0])
            return res
        return self._metric(primals, tangents)

    def left_sqrt_metric(self, primals, tangents):
        """Applies the left-square-root of the metric at `primals` to
        `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric.

        Returns
        -------
        metric_sample : tree-like structure
            Tree-like structure of the same type as primals to which the
            left-square-root of the metric has been applied to.
        """
        if self._left_sqrt_metric is None:
            nie = "`left_sqrt_metric` is not implemented"
            raise NotImplementedError(nie)
        return self._left_sqrt_metric(primals, tangents)

    def inv_metric(self, primals, tangents, cg=cg, **cg_kwargs):
        """Applies the inverse metric at `primals` to `tangents`.

        Parameters
        ----------
        primals : tree-like structure
            Position at which to evaluate the metric.
        tangents : tree-like structure
            Instance to which to apply the metric.
        cg : callable
            Implementation of the conjugate gradient algorithm and used to
            apply the inverse of the metric.
        cg_kwargs : dict
            Additional keyword arguments passed on to `cg`.

        Returns
        -------
        inv_naturally_curved : tree-like structure
            Tree-like structure of the same type as primals to which the
            inverse metric has been applied to.
        """
        res, _ = cg(Partial(self.metric, primals), tangents, **cg_kwargs)
        return res

    def draw_sample(
        self,
        primals,
        key,
        from_inverse: bool = False,
        cg: callable = cg,
        **cg_kwargs
    ):
        r"""Draws a sample of which the covariance is the metric
        (`from_inverse=False`) or the inverse metric (`from_inverse=True`).

        To sample from the inverse metric, we need to be able to draw samples
        which have the metric as covariance structure and we need to be able to
        apply the inverse metric. The first part is trivial since we can use
        the left square root of the metric :math:`L` associated with every
        likelihood:

        .. math::
            :nowrap:

            \begin{gather*}
                \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1}) \\
                t = L \tilde{d}
            \end{gather*}

        with :math:`t` now having a covariance structure of

        .. math::
            <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M .

        We now need to apply the inverse metric in order to transform the
        sample to an inverse sample. We can do so using the conjugate gradient
        algorithm which yields the solution to $M s = t$, i.e. applies the
        inverse of $M$ to $t$:

        .. math::
            :nowrap:

            \begin{gather*}
                M s =  t \\
                s = M^{-1} t = cg(M, t) .
            \end{gather*}

        Parameters
        ----------
        primals : tree-like structure
            Position at which to draw samples.
        key : tuple, list or np.ndarray of uint32 of length two
            Random key with which to generate random variables in data domain.
        from_inverse : bool
            Whether to draw samples from the metric or the inverse metric.
        cg : callable
            Implementation of the conjugate gradient algorithm and used to
            apply the inverse of the metric.
        cg_kwargs : dict
            Additional keyword arguments passed on to `cg`.

        Returns
        -------
        sample : tree-like structure
            Sample of which the covariance is the metric (`from_inverse=False`)
            or the inverse metric (`from_inverse=True`).
        """
        if self._lsm_tan_shp is None:
            nie = "Cannot draw sample without knowing the shape of the data"
            raise NotImplementedError(nie)

        if from_inverse:
            nie = "Cannot draw from the inverse of this operator"
            raise NotImplementedError(nie)
        else:
            white_sample = random_like_shapewdtype(self._lsm_tan_shp, key=key)
            return self.left_sqrt_metric(primals, white_sample)

    @property
    def left_sqrt_metric_tangents_shape(self):
        """Retrieves the shape of the tangent domain of the
        left-square-root of the metric.
        """
        return self._lsm_tan_shp

    def new(
        self, energy: callable, left_sqrt_metric: callable, metric: callable
    ):
        """Instantiates a new likelihood with the same `lsm_tangents_shape`.

        Parameters
        ----------
        energy : callable
            Function evaluating the negative log-likelihood.
        left_sqrt_metric : callable, optional
            Function applying the left-square-root of the metric.
        metric : callable, optional
            Function applying the metric.
        """
        return Likelihood(
            energy,
            left_sqrt_metric=left_sqrt_metric,
            metric=metric,
            lsm_tangents_shape=self._lsm_tan_shp
        )

    def jit(self):
        """Returns a new likelihood with jit-compiled energy, left-square-root
        of metric and metric.
        """
        from jax import jit

        if self._left_sqrt_metric is not None:
            j_lsm = jit(self._left_sqrt_metric)
            j_m = jit(self.metric)
        else:
            j_lsm = None
            j_m = None
        return self.new(
            jit(self._hamiltonian), left_sqrt_metric=j_lsm, metric=j_m
        )

    def __matmul__(self, f):
        def energy_at_f(primals):
            return self.energy(f(primals))

        def metric_at_f(primals, tangents):
            y, t = jvp(f, (primals, ), (tangents, ))
            r = self.metric(y, t)
            _, bwd = vjp(f, primals)
            res = bwd(r)
            return res[0]

        def left_sqrt_metric_at_f(primals, tangents):
            y, bwd = vjp(f, primals)
            left_at_fp = self.left_sqrt_metric(y, tangents)
            return bwd(left_at_fp)[0]

        return self.new(
            energy_at_f,
            left_sqrt_metric=left_sqrt_metric_at_f,
            metric=metric_at_f
        )

    def __add__(self, other):
        if not isinstance(other, Likelihood):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(other)!r}"
            )
            raise TypeError(te)

        def joined_hamiltonian(p):
            return self.energy(p) + other.energy(p)

        def joined_metric(p, t):
            return self.metric(p, t) + other.metric(p, t)

        joined_tangents_shape = {
            "lh_left": self._lsm_tan_shp,
            "lh_right": other._lsm_tan_shp
        }

        def joined_left_sqrt_metric(p, t):
            return self.left_sqrt_metric(
                p, t["lh_left"]
            ) + other.left_sqrt_metric(p, t["lh_right"])

        return Likelihood(
            joined_hamiltonian,
            left_sqrt_metric=joined_left_sqrt_metric,
            metric=joined_metric,
            lsm_tangents_shape=joined_tangents_shape
        )


class StandardHamiltonian(Likelihood):
    """Joined object storage composed of a user-defined likelihood and a
    standard normal likelihood as prior.
    """
    def __init__(self, likelihood: Likelihood, _compile_joined: bool = False):
        """Instantiates a new standardized Hamiltonian, i.e. a likelihood
        joined with a standard normal prior.

        Parameters
        ----------
        likelihood : Likelihood
            Energy, left-square-root of metric and metric of the likelihood.
        """
        self._nll = likelihood

        def joined_hamiltonian(primals):
            return self._nll(primals) + 0.5 * sum_of_squares(primals)

        def joined_metric(primals, tangents):
            return self._nll.metric(primals, tangents) + tangents

        if _compile_joined:
            from jax import jit
            joined_hamiltonian = jit(joined_hamiltonian)
            joined_metric = jit(joined_metric)
        self._hamiltonian = joined_hamiltonian
        self._metric = joined_metric
        # We do not know the shape of the tangent space of the left_sqrt_metric
        self._left_sqrt_metric = None
        self._lsm_tan_shp = None

    def jit(self):
        return StandardHamiltonian(self._nll.jit(), _compile_joined=True)

    def draw_sample(self, primals, key, from_inverse=False, cg=cg, **cg_kwargs):
        from jax import random
        subkey_nll, subkey_prr = random.split(key, 2)
        if from_inverse:
            nll_smpl = self._nll.draw_sample(primals, key=subkey_nll)
            prr_inv_metric_smpl = random_like(primals, key=subkey_prr)
            # One may transform any metric sample to a sample of the inverse
            # metric by simply applying the inverse metric to it
            prr_smpl = prr_inv_metric_smpl

            # Note, we can sample antithetically by swapping the global sign of
            # the metric sample below (which corresponds to mirroring the final
            # sample) and additionally by swapping the relative sign between the
            # prior and the likelihood sample. The first technique is
            # computationally cheap and empirically known to improve stability.
            # The latter technique requires an additional inversion and its
            # impact on stability is still unknown.
            # TODO: investigate the impact of sampling the prior and likelihood
            # antithetically.
            met_smpl = nll_smpl + prr_smpl
            # TODO: Set sensible convergence criteria
            """
            lambda x: met(pos, x),
            absdelta=1. / 100,
            resnorm=np.linalg.norm(met_smpl, ord=1) / 2,
            norm_ord=1
            """
            signal_smpl = self.inv_metric(
                primals, met_smpl, cg=cg, x0=prr_inv_metric_smpl, **cg_kwargs
            )
            return signal_smpl
        else:
            nll_smpl = self._nll.draw_sample(primals, key=subkey_nll)
            prr_inv_metric_smpl = random_like(primals, key=subkey_prr)
            return nll_smpl + prr_smpl, key
