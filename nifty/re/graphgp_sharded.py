"""graphgp Gaussian processes on a device mesh: sharded graphs and their generation.

The field has the user shape ``(n_points, ...)`` (any shape whose leading axis is
sharded), flattened in C order; shard ``s`` owns the contiguous block
``[s * B, (s + 1) * B)``. A :class:`ShardedGraph` holds static per-shard tables
that let every shard refine its own points level by level with two ingredients
that make it cheap:

* **compaction**: at every level only the points of that level are evaluated,
  padded to the level's maximum over the shards, instead of all local points;
* **halo exchange**: a shard keeps an *extended* local array
  ``[block B | halo H | initial n0 | junk]``. The halo holds the values of the
  other shards' points its own points condition on. A value is sent once, at
  the level it is generated: every shard writes the own values other shards
  need into a fixed export buffer, one ``all_gather`` per level, and a fixed
  gather fills the halo slots.

The levels are grouped into consecutive runs of similar size (power-of-two
classes), each run one ``lax.scan`` with static shapes. The derivatives are
custom: forward and reverse mode rebuild every level's input from the final
extended array (a value is final once generated), so nothing is stored per
level and the backward pass costs the memory of the forward pass.

Two ways to a :class:`ShardedGraph`:

* :func:`sharded_graph_from_graph`: the tables of a global ``graphgp.Graph``
  (built on one device) for ``n_shards`` blocks, host side;
* :func:`build_sharded_graph`: the graph is built *on the devices*. Every device
  builds the graph of its own block with graphgp (k-d tree, ``n0 / n_shards``
  initial points, ``k`` preceding neighbours in the block); the blocks are then
  merged on the mesh: a point of tree level ``L`` can only replace one of its
  ``k`` local predecessors by a foreign point closer than its ``k``-th local
  predecessor (``r_k``), so it is sent as a query to every shard whose points
  can lie within ``r_k`` (bounding boxes of the owner tree's subtrees, a
  superset); the owner answers with its ``k`` nearest points on tree levels
  strictly below ``L`` (graphgp's tree is stored level by level, so this is the
  prefix ``max_index = 2**L - 1`` of its own tree, the same restriction
  graphgp's preceding search uses) and closer than ``r_k`` (radius-limited
  search); every point keeps the ``k`` nearest of its local predecessors and
  the hits. "Preceding" is thus: earlier in the own tree, or on a lower tree
  level of a foreign tree (acyclic). The dependency depths are recomputed by
  relaxation with the foreign depths exchanged through the halo, and the
  tables are assembled on the devices; only a few counts (static sizes) pass
  through the host. The union of the shards' initial sets is the global dense
  set.

:class:`~nifty.re.gaussian_process.GraphGP` with ``mesh=`` uses this module.
"""
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import copy

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .logger import logger

__all__ = ["ShardedGraph", "sharded_graph_from_graph", "build_sharded_graph"]


@dataclass(frozen=True)
class ShardedGraph:
    """Static per-shard tables (leading axis: shard), see the module docstring.

    Extended array layout: ``[block B | halo H | initial n0 | junk]``,
    ``junk = B + H + n0`` is a scratch slot that padded entries read and write.
    Per level group ``g`` (``n_g`` consecutive levels, sizes ``M_g``/``E_g``/``I_g``):
    ``own`` (S, n_g, M_g) block positions of the level's points, ``nbr`` (S, n_g,
    M_g, k) their neighbours in the extended array, ``pts`` (S, n_g, M_g, d)
    their coordinates, ``valid`` (S, n_g, M_g) 1 for real points, ``exp`` (S,
    n_g, E_g) block positions of the values exported after the level,
    ``imp_src``/``imp_pos``/``imp_slot`` (S, n_g, I_g) where the gathered
    exports go (source shard, position, halo slot).
    """

    n_shards: int
    block_size: int
    halo_size: int
    n0: int
    n_levels: int
    group_sizes: Tuple[int, ...]  # levels per group, in level order
    own: Tuple[np.ndarray, ...]
    nbr: Tuple[np.ndarray, ...]
    pts: Tuple[np.ndarray, ...]
    valid: Tuple[np.ndarray, ...]
    exp: Tuple[np.ndarray, ...]
    imp_src: Tuple[np.ndarray, ...]
    imp_pos: Tuple[np.ndarray, ...]
    imp_slot: Tuple[np.ndarray, ...]
    lvl: Tuple[np.ndarray, ...]  # (S, n_g) level number of every step of the group
    ext_points: np.ndarray  # (S, junk + 1, d)
    ext_level: np.ndarray  # (S, junk + 1) generation level of every extended entry (junk: n_levels)
    init_own: np.ndarray  # (S, n0_max) block positions of the shard's initial points (padded with junk)
    init_slot_own: np.ndarray  # (S, n0_max) their slots in the initial set
    init_src: np.ndarray  # (n0,) shard owning each initial point
    init_pos: np.ndarray  # (n0,) its position in that shard's init_own
    initial_points: np.ndarray  # (n0, d)
    halo: np.ndarray  # (S, H) global user indices of the halo entries (-1 padding), for tests

    @property
    def junk(self) -> int:
        return self.block_size + self.halo_size + self.n0

    @property
    def n_groups(self) -> int:
        return len(self.group_sizes)


jax.tree_util.register_dataclass(
    ShardedGraph,
    data_fields=["own", "nbr", "pts", "valid", "exp", "imp_src", "imp_pos", "imp_slot", "lvl", "ext_points", "ext_level", "init_own", "init_slot_own", "init_src", "init_pos", "initial_points", "halo"],
    meta_fields=["n_shards", "block_size", "halo_size", "n0", "n_levels", "group_sizes"],
)


def _pad_rows(rows, width, fill):
    out = np.full((len(rows), width), fill, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def sharded_graph_from_graph(graph, n_shards: int, n_points: Optional[int] = None, *, group_by="log2") -> ShardedGraph:
    """The tables of :class:`ShardedGraph` for a global ``graphgp.Graph`` (with
    ``indices``, the user order) on ``n_shards`` contiguous blocks (host side)."""
    points = np.asarray(graph.points, dtype=np.float32)
    neighbors = np.asarray(graph.neighbors)
    offsets = tuple(int(o) for o in graph.offsets)
    if graph.indices is None:
        raise ValueError("the graph needs `indices` (user order of the points)")
    user_of_tree = np.asarray(graph.indices)
    n = points.shape[0] if n_points is None else int(n_points)
    if n % n_shards != 0:
        raise ValueError(f"{n} points are not divisible into {n_shards} shards")
    B = n // n_shards
    n0 = offsets[0]
    k = neighbors.shape[1]
    d = points.shape[1]
    L = len(offsets)

    level = np.zeros(n, dtype=np.int64)
    level[user_of_tree] = np.searchsorted(np.asarray(offsets), np.arange(n), side="right")  # tree level (batch) of every tree point
    nbr = np.full((n, k), -1, dtype=np.int64)
    nbr[user_of_tree[n0:]] = user_of_tree[neighbors]
    pts = np.empty_like(points)
    pts[user_of_tree] = points
    initial_index = user_of_tree[:n0]
    init_slot = np.full(n, -1, dtype=np.int64)
    init_slot[initial_index] = np.arange(n0)
    owner = np.arange(n) // B

    # halos: foreign, non-initial predecessors of each shard's points
    halos = []
    for s in range(n_shards):
        nb = nbr[s * B : (s + 1) * B]
        nb = nb[nb >= 0]
        halos.append(np.unique(nb[(owner[nb] != s) & (init_slot[nb] < 0)]))
    H = max(max((len(h) for h in halos), default=0), 1)  # XLA needs non-empty arrays
    junk = B + H + n0
    halo_pos = np.full((n_shards, n), -1, dtype=np.int64)
    for s, h in enumerate(halos):
        halo_pos[s, h] = np.arange(len(h))

    def ext_index(s, g):
        g = np.asarray(g)
        return np.where(init_slot[g] >= 0, B + H + init_slot[g], np.where(owner[g] == s, g - s * B, B + halo_pos[s, g]))

    # exports: for every owner s and level l the own points some other shard needs
    needed = [[set() for _ in range(n_shards)] for _ in range(L)]  # needed[l][s]: global indices
    for t, h in enumerate(halos):
        for g in h:
            needed[level[g]][owner[g]].add(int(g))
    exp_lists = [[np.array(sorted(needed[l][s]), dtype=np.int64) for s in range(n_shards)] for l in range(L)]
    exp_pos = {}
    for l in range(L):
        for s in range(n_shards):
            for i, g in enumerate(exp_lists[l][s]):
                exp_pos[int(g)] = i

    # per level tables
    own_l, nbr_l, pts_l, valid_l, exp_l, imp_l = [], [], [], [], [], []
    M_l, E_l, I_l = [], [], []
    for l in range(1, L):
        own_rows, exp_rows, imp_rows = [], [], []
        for s in range(n_shards):
            blk = np.arange(s * B, (s + 1) * B)
            own_rows.append(np.where(level[blk] == l)[0])
            exp_rows.append(exp_lists[l][s] - s * B)
            h = halos[s]
            h = h[level[h] == l]
            imp_rows.append(np.stack([owner[h], np.array([exp_pos[int(g)] for g in h], dtype=np.int64), B + halo_pos[s, h]], axis=1) if h.size else np.zeros((0, 3), dtype=np.int64))
        M = max(len(r) for r in own_rows)
        E = max(len(r) for r in exp_rows)
        I = max(len(r) for r in imp_rows)
        M_l.append(M), E_l.append(E), I_l.append(I)
        own = _pad_rows(own_rows, M, junk)
        valid = np.zeros((n_shards, M), dtype=np.float64)
        nb_ext = np.full((n_shards, M, k), junk, dtype=np.int64)
        pt = np.zeros((n_shards, M, d), dtype=np.float32)
        for s, r in enumerate(own_rows):
            valid[s, : len(r)] = 1.0
            g = r + s * B
            nb_ext[s, : len(r)] = ext_index(s, nbr[g])
            pt[s, : len(r)] = pts[g]
        imp = np.stack([_pad_rows([r[:, j] for r in imp_rows], I, 0 if j < 2 else junk) for j in range(3)], axis=0)
        own_l.append(own), nbr_l.append(nb_ext), pts_l.append(pt), valid_l.append(valid)
        exp_l.append(_pad_rows(exp_rows, E, junk)), imp_l.append(imp)

    # groups of consecutive levels with similar size
    if group_by == "log2":
        cls = [int(np.ceil(np.log2(max(m, 1)))) for m in M_l]
    else:
        cls = [0] * len(M_l)
    group_sizes = []
    for i, c in enumerate(cls):
        if i > 0 and c == cls[i - 1]:
            group_sizes[-1] += 1
        else:
            group_sizes.append(1)

    def stack(tables, start, size, width_pad, fill, trailing=()):
        w = max(width_pad[start : start + size])
        out = np.full((n_shards, size) + (w,) + trailing, fill, dtype=tables[start].dtype)
        for i in range(size):
            t = tables[start + i]
            out[:, i, : t.shape[1]] = t
        return out

    own_g, nbr_g, pts_g, valid_g, exp_g, src_g, pos_g, slot_g, lvl_g = ([] for _ in range(9))
    start = 0
    for size in group_sizes:
        lvl_g.append(np.broadcast_to(np.arange(start + 1, start + size + 1, dtype=np.int64), (n_shards, size)).copy())
        own_g.append(stack(own_l, start, size, M_l, junk))
        nbr_g.append(stack(nbr_l, start, size, M_l, junk, (k,)))
        pts_g.append(stack(pts_l, start, size, M_l, 0.0, (d,)))
        valid_g.append(stack(valid_l, start, size, M_l, 0.0))
        exp_g.append(stack(exp_l, start, size, E_l, junk))
        src_g.append(stack([t[0] for t in imp_l], start, size, I_l, 0))
        pos_g.append(stack([t[1] for t in imp_l], start, size, I_l, 0))
        slot_g.append(stack([t[2] for t in imp_l], start, size, I_l, junk))
        start += size

    # extended coordinates and generation levels
    ext_points = np.zeros((n_shards, junk + 1, d), dtype=np.float32)
    ext_level = np.full((n_shards, junk + 1), L, dtype=np.int64)
    halo_tab = np.full((n_shards, H), -1, dtype=np.int64)
    for s in range(n_shards):
        blk = np.arange(s * B, (s + 1) * B)
        ext_points[s, :B] = pts[blk]
        ext_level[s, :B] = level[blk]
        h = halos[s]
        ext_points[s, B : B + len(h)] = pts[h]
        ext_level[s, B : B + len(h)] = level[h]
        halo_tab[s, : len(h)] = h
        ext_points[s, B + H : B + H + n0] = pts[initial_index]
        ext_level[s, B + H : B + H + n0] = 0
    # initial points: each shard exports the excitations of its own ones
    init_rows = [np.where(init_slot[s * B : (s + 1) * B] >= 0)[0] for s in range(n_shards)]
    n0_max = max(len(r) for r in init_rows)
    init_own = _pad_rows(init_rows, n0_max, junk)
    init_slot_own = _pad_rows([init_slot[r + s * B] for s, r in enumerate(init_rows)], n0_max, 0)
    init_src = owner[initial_index]
    init_pos = np.array([int(np.where(init_rows[owner[g]] == g - owner[g] * B)[0][0]) for g in initial_index], dtype=np.int64)

    return ShardedGraph(
        n_shards, B, H, n0, L, tuple(group_sizes),
        tuple(own_g), tuple(nbr_g), tuple(pts_g), tuple(valid_g), tuple(exp_g), tuple(src_g), tuple(pos_g), tuple(slot_g),
        tuple(lvl_g), ext_points, ext_level, init_own, init_slot_own, init_src, init_pos, pts[initial_index], halo_tab,
    )


# ---------------------------------------------------------------------------- the device build
def _query_radius_jax(points, split_dims, queries, max_indices, square_radii, *, k):
    """Radius-limited k-nearest search of external `queries` with graphgp's
    tree traversal (``graphgp.tree._traverse_tree`` takes the initial square
    radius; the update rule is graphgp's ``_single_query_neighbors``). Pure
    JAX; ``-1`` for unfilled slots. Fallback for graphgp versions whose
    ``query_neighbors`` has no ``max_square_distances``."""
    from graphgp.tree import _traverse_tree

    n = points.shape[0]
    ext = jnp.concatenate([points, queries.astype(points.dtype)], axis=0)
    split_ext = jnp.concatenate([split_dims, jnp.zeros(queries.shape[0], split_dims.dtype)])

    def single(query_index, max_index, r2):
        def update_func(node, state, _):
            neighbors, sq = state
            d2 = jnp.sum(jnp.square(ext[query_index] - ext[node]))
            m = jnp.argmax(sq)
            neighbors, sq = lax.cond(
                d2 < sq[m], lambda _: (neighbors.at[m].set(node), sq.at[m].set(d2)), lambda _: (neighbors, sq), None
            )
            return (neighbors, sq), jnp.max(sq)

        state = (-jnp.ones(k, dtype=jnp.int32), jnp.full(k, r2, ext.dtype))
        (neighbors, _) = _traverse_tree(ext, split_ext, query_index, max_index, update_func, state, r2)
        return neighbors

    query_indices = n + jnp.arange(queries.shape[0], dtype=jnp.int32)
    with jax.enable_x64(False):  # graphgp's traversal mixes 32 and 64 bit under x64
        return jax.vmap(single)(query_indices, max_indices.astype(jnp.int32), square_radii.astype(points.dtype))


def _query_points(points, split_dims, queries, max_indices, radii, *, k, cuda=False):
    """The ``k`` nearest tree points with position below `max_indices` and
    closer than `radii` of the external `queries` (coordinates, ``(Q, d)``):
    the queries are appended to the tree points, the traversal only ever
    visits nodes below `max_indices` (at most the tree size), so the appended
    rows are query coordinates only. The radius makes queries that cannot hit
    anything cheap (the unbounded search of a far query sweeps the owner's
    boundary region). Uses ``graphgp.tree.query_neighbors(...,
    max_square_distances)`` (CUDA kernel with `cuda`, else pure JAX; about 8x
    slower, 0.17 s per 2^18 queries on an A100); with an older graphgp the
    radius search runs on graphgp's traversal directly. Returns tree positions
    (``-1`` where fewer exist) and distances, sorted by distance."""
    import inspect

    from graphgp.tree import query_neighbors

    r2 = jnp.square(radii).astype(points.dtype)
    if "max_square_distances" in inspect.signature(query_neighbors).parameters:
        n = points.shape[0]
        ext = jnp.concatenate([points, queries.astype(points.dtype)], axis=0)
        split_ext = jnp.concatenate([split_dims, jnp.zeros(queries.shape[0], split_dims.dtype)])
        query_indices = n + jnp.arange(queries.shape[0], dtype=split_dims.dtype)
        if cuda:
            hits = query_neighbors(ext, split_ext, query_indices, max_indices.astype(split_dims.dtype), r2, k=k, cuda=True)
        else:
            with jax.enable_x64(False):  # graphgp's pure-JAX traversal mixes 32 and 64 bit under x64
                hits = query_neighbors(ext, split_ext, query_indices.astype(jnp.int32), max_indices.astype(jnp.int32), r2, k=k, cuda=False)
    else:  # graphgp without the radius argument: pure JAX on graphgp's traversal
        hits = _query_radius_jax(points, split_dims, queries, max_indices, r2, k=k)
    hits = jnp.asarray(hits).astype(jnp.int32)
    # unfilled slots are -1; the root is visited even for max_index 0
    valid = (hits >= 0) & (hits < max_indices[:, None])
    hits = jnp.where(valid, hits, -1)
    d2 = jnp.sum(jnp.square(points[jnp.maximum(hits, 0)].astype(jnp.float64) - queries[:, None, :].astype(jnp.float64)), axis=-1)
    dist = jnp.where(valid, jnp.sqrt(d2), jnp.inf).astype(jnp.float32)
    dist, hits = lax.sort((dist, hits), dimension=1, num_keys=2)
    return hits, dist


def _tree_level(pos):
    """Level of a tree position in graphgp's layout (level l at [2^l - 1, 2^(l+1) - 1)):
    ``floor(log2(pos + 1))`` in 32-bit integer arithmetic (positions are below 2^31;
    64-bit shifts would silently overflow when x64 is disabled)."""
    pos = jnp.asarray(pos).astype(jnp.int32)[..., None] + 1
    thresholds = jnp.left_shift(jnp.int32(1), jnp.arange(31, dtype=jnp.int32))  # 2^0 .. 2^30
    return (jnp.sum(pos >= thresholds, axis=-1) - 1).astype(jnp.int32)


# ---------------------------------------------------------------------------- per-device local graphs
def _local_graph(pts, n0_s: int, k: int, *, cuda: bool):
    """graphgp's tree and preceding neighbours of one block (on its device):
    tree points, split dims, tree position -> block position, neighbours in
    tree positions (dummies for the initial points) and ``r_k``."""
    from graphgp.tree import build_tree, query_preceding_neighbors

    pts = jnp.asarray(pts, dtype=jnp.float32)
    if cuda:
        pts_t, split_dims, idx = build_tree(pts, cuda=True)
        nbr = query_preceding_neighbors(pts_t, split_dims, n0=n0_s, k=k, cuda=True)
    else:
        with jax.enable_x64(False):  # graphgp's pure-JAX tree builder needs 32 bit
            pts_t, split_dims, idx = build_tree(pts, cuda=False)
            nbr = query_preceding_neighbors(pts_t, split_dims, n0=n0_s, k=k, cuda=False)
    if isinstance(nbr, tuple):
        nbr = nbr[0]
    nbr = jnp.concatenate([jnp.zeros((n0_s, k), jnp.int32), jnp.asarray(nbr).astype(jnp.int32)], axis=0)
    ptsd = pts_t.astype(jnp.float64)
    d = jnp.linalg.norm(ptsd[nbr] - ptsd[:, None, :], axis=-1)
    r_k = jnp.where(jnp.arange(pts.shape[0]) < n0_s, 0.0, d.max(axis=1))
    return pts_t.astype(jnp.float32), split_dims.astype(jnp.int32), idx.astype(jnp.int32), nbr, r_k


def _stack(mesh, arrays):
    """One sharded array ``(S, ...)`` from per-device arrays (no host round trip)."""
    arrays = [jax.device_put(a[None], d) for a, d in zip(arrays, mesh.devices.ravel())]
    shape = (len(arrays),) + arrays[0].shape[1:]
    return jax.make_array_from_single_device_arrays(shape, NamedSharding(mesh, P(mesh.axis_names[0])), arrays)


# ---------------------------------------------------------------------------- the device build
class _DeviceGraph:
    """Result of :func:`_build_device_graph`, per shard (leading axis ``S``),
    in the block's user order: ``neighbors`` (S, B, k) global user ids of the
    merged neighbours (``-1`` for the initial points), ``initial`` (S, B) mask
    of the initial points, ``depth`` (S, B) dependency depths (0 = initial),
    ``tree_pos`` (S, B) position in the block's own tree; ``points`` (N, d)
    in user order; ``halo`` (S, H) sorted global ids of the foreign neighbours
    (``N`` padding) with their depths ``halo_depth``."""

    def __init__(self, points, neighbors, initial, depth, tree_pos, halo, halo_depth, *, n0, k, stats):
        self.points = points
        self.neighbors, self.initial, self.depth, self.tree_pos = neighbors, initial, depth, tree_pos
        self.halo, self.halo_depth = halo, halo_depth
        self.n0, self.k, self.stats = n0, k, stats

    @property
    def n_shards(self):
        return self.neighbors.shape[0]

    @property
    def block_size(self):
        return self.neighbors.shape[1]


def _build_device_graph(points, mesh: Mesh, *, n0: int, k: int, cuda: bool = True, verbose: bool = False) -> _DeviceGraph:
    """Build the sharded graph of `points` (``(N, d)`` in user order, split into
    contiguous blocks over the one-dimensional `mesh`) on the devices; see the
    module docstring. `n0` is the global number of initial points (``n0 //
    n_shards`` per block)."""
    ax = mesh.axis_names[0]
    S = mesh.shape[ax]
    points = np.asarray(points, dtype=np.float32).reshape(-1, np.shape(points)[-1])
    N, d = points.shape
    if N % S:
        raise ValueError(f"{N} points are not divisible into {S} shards")
    B = N // S
    n0_s = n0 // S
    if n0_s < k:
        raise ValueError(f"n0 / n_shards = {n0_s} must be at least k = {k}")
    import time

    t0 = time.time()
    log = (lambda msg: logger.info(f"sharded graph: {msg} ({time.time() - t0:.1f} s)")) if verbose else (lambda msg: None)
    devices = mesh.devices.ravel()
    local = [_local_graph(jax.device_put(points[s * B : (s + 1) * B], dev), n0_s, k, cuda=cuda) for s, dev in enumerate(devices)]
    pts_t = _stack(mesh, [l[0] for l in local])  # (S, B, d) tree order
    split_dims = _stack(mesh, [l[1] for l in local])
    idx = _stack(mesh, [l[2] for l in local])  # tree position -> block position
    nbr_loc = _stack(mesh, [l[3] for l in local])  # (S, B, k) tree positions
    r_k = _stack(mesh, [l[4] for l in local])
    jax.block_until_ready(r_k)
    log(f"{S} local graphs of {B} points (n0 {n0_s}, k {k})")
    smap = partial(shard_map, mesh=mesh, check_vma=False)

    # subtree boxes: the owner tree's level-`box_level` subtrees (points of upper levels individually)
    box_level = max(0, min(6, int(np.floor(np.log2(B + 1))) - 3))
    n_box = 2**box_level
    n_upper = n_box - 1
    level = _tree_level(jnp.arange(B))

    def _need(pts_t, r_k, sidx):
        """(S, B) which of my points must be queried against which shard: a
        foreign point can only enter the k nearest if it is closer than
        ``r_k(p)``, tested against the bounding boxes of the owner tree's
        level-`box_level` subtrees (a lower bound, so a superset)."""
        pos = jnp.arange(B)
        anc = jnp.where(level >= box_level, ((pos + 1) >> jnp.maximum(level - box_level, 0)) - 1 - n_upper, n_box)  # subtree id, n_box: dropped
        lo = jax.ops.segment_min(pts_t, anc, num_segments=n_box)
        hi = jax.ops.segment_max(pts_t, anc, num_segments=n_box)
        boxes = lax.all_gather(jnp.stack([lo, hi], axis=1), ax, axis=0)  # (S, n_box, 2, d)
        upper = lax.all_gather(pts_t[:n_upper], ax, axis=0)  # (S, n_upper, d)

        def min_dist(t):
            def box(i, m):
                gap = jnp.maximum(jnp.maximum(boxes[t, i, 0] - pts_t, pts_t - boxes[t, i, 1]), 0.0)
                return jnp.minimum(m, jnp.sum(jnp.square(gap), axis=-1))

            def point(i, m):
                return jnp.minimum(m, jnp.sum(jnp.square(upper[t, i] - pts_t), axis=-1))

            m = lax.fori_loop(0, n_box, box, jnp.full((B,), jnp.inf, pts_t.dtype))
            return lax.fori_loop(0, n_upper, point, m)

        d2 = jnp.stack([min_dist(t) for t in range(S)])  # (S, B)
        return (d2 < jnp.square(r_k[None, :]).astype(pts_t.dtype)) & (jnp.arange(S)[:, None] != sidx) & (jnp.arange(B)[None, :] >= n0_s)

    # ---- phase 1: number of queries per target (static buffer size)
    def count_queries(pts_t, r_k):
        need = _need(pts_t[0], r_k[0], lax.axis_index(ax))
        return need.sum(axis=1)[None]

    counts = np.asarray(smap(count_queries, in_specs=(P(ax), P(ax)), out_specs=P(ax))(pts_t, r_k))
    E = max(int(counts.max()), 1)
    log(f"queries per target up to {E} ({100 * counts.sum() / N:.0f} % of the points, summed over targets; {n_box} subtree boxes)")
    chunk = 2**18

    # ---- phase 2: send the queries, answer on graphgp's tree, send the hits back, merge
    def merge(pts_t, split_dims, idx, nbr_loc, r_k):
        pts_t, split_dims, idx, nbr_loc, r_k = pts_t[0], split_dims[0], idx[0], nbr_loc[0], r_k[0]
        sidx = lax.axis_index(ax)
        need = _need(pts_t, r_k, sidx)
        slots = jax.vmap(lambda n: jnp.nonzero(n, size=E, fill_value=-1)[0])(need)  # (S_tgt, E) tree positions
        valid = slots >= 0
        sl = jnp.where(valid, slots, 0)
        coords = jnp.where(valid[..., None], pts_t[sl], jnp.inf)
        payload = jnp.concatenate([coords, level[sl][..., None].astype(pts_t.dtype), r_k[sl][..., None].astype(pts_t.dtype)], axis=-1)  # (S_tgt, E, d + 2) float32
        recv = lax.all_to_all(payload, ax, 0, 0, tiled=False)  # (S_src, E, d + 1): queries to me

        def answer_chunk(args):
            q, qlevel, qr = args
            qvalid = jnp.isfinite(q[:, 0])
            max_index = jnp.where(qvalid, jnp.minimum((1 << jnp.maximum(qlevel, 0)) - 1, B), 0)  # levels strictly below L
            hits, dist = _query_points(pts_t, split_dims, jnp.where(qvalid[:, None], q, 0.0), max_index, jnp.where(qvalid, qr, 0.0), k=k, cuda=cuda)
            hit_ids = jnp.where(hits >= 0, sidx * B + idx[jnp.maximum(hits, 0)], -1).astype(jnp.int32)  # global user ids
            return jnp.concatenate([hit_ids, lax.bitcast_convert_type(dist.astype(jnp.float32), jnp.int32)], axis=1)

        q = recv[..., :d].reshape(-1, d)
        qlevel = recv[..., d].reshape(-1).astype(jnp.int32)
        qr = recv[..., d + 1].reshape(-1)
        n_q = q.shape[0]
        pad = (-n_q) % chunk
        q = jnp.concatenate([q, jnp.full((pad, d), jnp.inf, q.dtype)])
        qlevel = jnp.concatenate([qlevel, jnp.zeros(pad, jnp.int32)])
        qr = jnp.concatenate([qr, jnp.zeros(pad, qr.dtype)])
        answer = lax.map(answer_chunk, (q.reshape(-1, chunk, d), qlevel.reshape(-1, chunk), qr.reshape(-1, chunk)))
        answer = answer.reshape(-1, 2 * k)[:n_q].reshape(S, E, 2 * k)
        back = lax.all_to_all(answer, ax, 0, 0, tiled=False)  # (S_target, E, 2k): hits of my queries
        # merge target by target with a running best k (no (B, S, k) candidate arrays: 28 M points OOM'd)
        best_ids = (sidx * B + idx[nbr_loc]).astype(jnp.int32)
        best_dist = jnp.sqrt(jnp.sum(jnp.square(pts_t[nbr_loc].astype(jnp.float64) - pts_t[:, None, :].astype(jnp.float64)), axis=-1)).astype(jnp.float32)
        for t in range(S):
            row = jnp.where(valid[t], slots[t], B)  # B: dropped
            ids_t = jnp.full((B, k), -1, dtype=jnp.int32).at[row].set(back[t, :, :k], mode="drop")
            dist_t = jnp.full((B, k), jnp.inf, dtype=jnp.float32).at[row].set(lax.bitcast_convert_type(back[t, :, k:], jnp.float32), mode="drop")
            all_ids = jnp.concatenate([best_ids, ids_t], axis=1)
            all_dist = jnp.where(all_ids >= 0, jnp.concatenate([best_dist, dist_t], axis=1), jnp.inf)
            neg, sel = lax.top_k(-all_dist, k)
            best_ids, best_dist = jnp.take_along_axis(all_ids, sel, axis=1), -neg
        merged = best_ids
        is_init = jnp.arange(B) < n0_s
        merged = jnp.where(is_init[:, None], -1, merged)
        n_foreign = ((merged >= 0) & (merged // B != sidx)).sum()
        # back to the block's user order
        nbr_user = jnp.zeros((B, k), jnp.int32).at[idx].set(merged)
        init_user = jnp.zeros((B,), bool).at[idx].set(is_init)
        tree_pos = jnp.zeros((B,), jnp.int32).at[idx].set(jnp.arange(B, dtype=jnp.int32))
        return nbr_user[None], init_user[None], tree_pos[None], jnp.array([n_foreign])

    nbr_user, init_user, tree_pos, n_foreign = smap(merge, in_specs=(P(ax),) * 5, out_specs=(P(ax),) * 4)(pts_t, split_dims, idx, nbr_loc, r_k)
    jax.block_until_ready(nbr_user)  # (S, B, k) global user ids of the merged neighbours (-1 for initial points), user order
    log("queries answered and merged")

    # ---- phase 3: halo (counting pass, then static) and the depths
    def count_halo(nbr_user):
        ids = nbr_user[0].reshape(-1)
        foreign = (ids >= 0) & (ids // B != lax.axis_index(ax))
        srt = jnp.sort(jnp.where(foreign, ids, -1))
        distinct = (srt >= 0) & jnp.concatenate([jnp.array([True]), srt[1:] != srt[:-1]])
        return distinct.sum()[None]

    H = max(int(np.asarray(smap(count_halo, in_specs=P(ax), out_specs=P(ax))(nbr_user)).max()), 1)
    log(f"halo up to {H}")

    def depths(nbr_user, init_user):
        nbr, init = nbr_user[0], init_user[0]
        sidx = lax.axis_index(ax)
        ids = nbr.reshape(-1)
        ht = _halo_tables(ids, sidx, N=N, B=B, H=H, S=S, ax=ax)
        ext_pos = jnp.where(ht.foreign, B + jnp.searchsorted(ht.halo, ids), jnp.where(ids >= 0, ids - sidx * B, B + H)).reshape(B, k)

        def body(carry):
            depth, _, it = carry
            ext_depth = jnp.concatenate([depth, ht.gather(depth, -1), jnp.array([-1], jnp.int32)])
            new = jnp.where(init, 0, 1 + ext_depth[ext_pos].max(axis=1)).astype(jnp.int32)
            changed = lax.psum((new != depth).any().astype(jnp.int32), ax) > 0
            return new, changed, it + 1

        depth0 = jnp.where(init, 0, 1).astype(jnp.int32)
        depth, _, it = lax.while_loop(lambda c: c[1], body, (depth0, jnp.array(True), jnp.array(0)))
        return depth[None], ht.halo[None], ht.gather(depth, -1)[None], it[None]

    depth, halo, halo_depth, its = smap(depths, in_specs=(P(ax), P(ax)), out_specs=(P(ax),) * 4)(nbr_user, init_user)
    stats = dict(
        n0=S * n0_s, queries_max=E, queries=counts.tolist(), halo_max=H,
        foreign_neighbours=float(np.asarray(n_foreign).sum() / (N * k)), depth_iterations=int(np.asarray(its).max()),
        depth_max=int(np.asarray(depth).max()),
    )
    log(
        f"{N} points, {S} shards, n0 {stats['n0']}, foreign neighbours {100 * stats['foreign_neighbours']:.1f} %, "
        f"depth {stats['depth_max']} ({stats['depth_iterations']} relaxation sweeps)"
    )
    return _DeviceGraph(points, nbr_user, init_user, depth, tree_pos, halo, halo_depth, n0=S * n0_s, k=k, stats=stats)


class _HaloTables:
    """Halo of one shard inside a ``shard_map`` body: the sorted foreign ids
    (``N`` padding) among `ids`, the export list the owners need, and
    ``gather(attr, fill)`` fetching an owner attribute (``(B, ...)`` per
    shard) for every halo entry with one ``all_gather`` of the exports."""

    def __init__(self, ids, sidx, *, N, B, H, S, ax):
        self.foreign = (ids >= 0) & (ids // B != sidx)
        self.halo = jnp.unique(jnp.where(self.foreign, ids, N), size=H + 1, fill_value=N)[:H]
        self.halo_valid = self.halo < N
        needed = jnp.stack([jnp.sort(jnp.where(self.halo_valid & (self.halo // B == t), self.halo, N)) for t in range(S)])
        req = lax.all_to_all(needed, ax, 0, 0, tiled=False)  # (S_src, H): my ids that src needs
        n_exp = max(min(S * H, B), 1)
        self.exp_all = jnp.unique(req.reshape(-1), size=n_exp + 1, fill_value=N)[:n_exp]  # sorted global ids
        self.exp_valid = self.exp_all < N
        self.exp_loc = jnp.where(self.exp_valid, self.exp_all - sidx * B, 0)
        exp_lists = lax.all_gather(self.exp_all, ax, axis=0, tiled=False)  # (S, n_exp)
        self.imp_src = jnp.where(self.halo_valid, self.halo // B, 0)
        # position in the owner's export list: one searchsorted per owner (no (H, n_exp) gather)
        pos = jnp.zeros_like(self.halo)
        for t in range(S):
            pos = jnp.where(self.imp_src == t, jnp.searchsorted(exp_lists[t], self.halo), pos)
        self.imp_pos = jnp.where(self.halo_valid, pos, 0)
        self.ax = ax

    def gather(self, attr, fill):
        """`attr` (B, ...) of the owners for every halo entry."""
        return self.gather_exports(attr[self.exp_loc], fill)

    def gather_exports(self, buf, fill):
        """`buf` (n_exp, ...) in the owners' export order for every halo entry."""
        extra = (1,) * (buf.ndim - 1)
        buf = jnp.where(self.exp_valid.reshape(-1, *extra), buf, fill)
        g = lax.all_gather(buf, self.ax, axis=0, tiled=False)
        return jnp.where(self.halo_valid.reshape(-1, *extra), g[self.imp_src, self.imp_pos], fill)


_halo_tables = _HaloTables


def _device_graph_to_graph(dg: _DeviceGraph):
    """The global ``graphgp.Graph`` equivalent to a :class:`_DeviceGraph`
    (host): depth-major order (depth, shard, own tree position), ``offsets``
    from the depth counts, ``indices`` the user ids."""
    S, B = dg.n_shards, dg.block_size
    N = S * B
    depth = np.asarray(dg.depth).reshape(N)
    tree_pos = np.asarray(dg.tree_pos).reshape(N)
    nbr = np.asarray(dg.neighbors).reshape(N, dg.k)
    shard = np.arange(N) // B
    order = np.lexsort((tree_pos, shard, depth))  # global position -> user id
    pos = np.empty(N, dtype=np.int64)
    pos[order] = np.arange(N)
    counts = np.bincount(depth, minlength=depth.max() + 1)
    offsets = tuple(int(o) for o in np.cumsum(counts))
    n0 = offsets[0]
    neighbors = pos[nbr[order[n0:]]]
    import graphgp as ggp

    return ggp.Graph(jnp.asarray(dg.points[order], dtype=jnp.float32), jnp.asarray(neighbors, dtype=jnp.int32), offsets, jnp.asarray(order, dtype=jnp.int32))


# ---------------------------------------------------------------------------- compact tables on the devices
def _compact_tables(dg: _DeviceGraph, mesh: Mesh, *, group_by="log2", verbose: bool = False) -> ShardedGraph:
    """The :class:`ShardedGraph` tables of a
    :class:`_DeviceGraph`, assembled on the devices (same conventions as
    :func:`sharded_graph_from_graph` on :func:`_device_graph_to_graph`: halo and
    export lists sorted by global id, initial set ordered by shard and tree
    position). Only the per-level counts pass through the host (static sizes)."""
    import time

    t0 = time.time()
    ax = mesh.axis_names[0]
    S, B, k = dg.n_shards, dg.block_size, dg.k
    N = S * B
    n0_s = dg.n0 // S
    d = dg.points.shape[1]
    L = dg.stats["depth_max"] + 1  # levels 0 .. L-1 (0: initial)
    H_all = dg.halo.shape[1]  # foreign neighbours including initial ones
    smap = partial(shard_map, mesh=mesh, check_vma=False)
    pts_all = jax.device_put(dg.points.reshape(S, B, d), NamedSharding(mesh, P(ax)))

    def common(nbr, init, depth, tree_pos, sidx, H2):
        """Halo (size `H2`) of the non-initial foreign neighbours, their depths
        and export ranks, and the extended index of every neighbour slot."""
        junk = B + H2 + dg.n0
        ids = nbr.reshape(-1)
        h1 = _halo_tables(ids, sidx, N=N, B=B, H=H_all, S=S, ax=ax)
        tpos1 = h1.gather(tree_pos, B)  # tree position of every foreign neighbour (initial iff < n0_s)
        nbr_tpos = jnp.where(h1.foreign, tpos1[jnp.searchsorted(h1.halo, ids)], tree_pos[jnp.where(ids >= 0, ids - sidx * B, 0)])
        nbr_init = (ids >= 0) & (nbr_tpos < n0_s)
        h2 = _halo_tables(jnp.where(nbr_init, -1, ids), sidx, N=N, B=B, H=H2, S=S, ax=ax)
        halo_depth = h2.gather(depth, L)
        # rank of every export inside its level's (id-sorted) export list
        exp_depth = jnp.where(h2.exp_valid, depth[h2.exp_loc], L)
        order = jnp.argsort(exp_depth, stable=True)
        start = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(jnp.bincount(exp_depth, length=L + 1))[:-1].astype(jnp.int32)])
        rank = jnp.zeros_like(exp_depth).at[order].set(jnp.arange(exp_depth.shape[0], dtype=jnp.int32) - start[exp_depth[order]])
        halo_rank = h2.gather_exports(rank, 0)
        ext_idx = jnp.where(
            nbr_init, B + H2 + (ids // B) * n0_s + nbr_tpos,
            jnp.where(h2.foreign, B + jnp.searchsorted(h2.halo, ids), jnp.where(ids >= 0, ids - sidx * B, junk)),
        ).reshape(B, k)
        return h2, halo_depth, exp_depth, halo_rank, ext_idx

    def count(nbr_user, init_user, depth, tree_pos):
        h2, halo_depth, exp_depth, _, _ = common(nbr_user[0], init_user[0], depth[0], tree_pos[0], lax.axis_index(ax), H_all)
        M = jnp.bincount(depth[0], length=L)
        E = jnp.bincount(exp_depth, length=L + 1)[:L]
        I = jnp.bincount(jnp.where(h2.halo_valid, halo_depth, L), length=L + 1)[:L]
        return M[None], E[None], I[None], h2.halo_valid.sum()[None]

    M, E, I, H2 = (np.asarray(a).max(axis=0) for a in smap(count, in_specs=(P(ax),) * 4, out_specs=(P(ax),) * 4)(dg.neighbors, dg.initial, dg.depth, dg.tree_pos))
    H = max(int(H2), 1)  # non-initial foreign neighbours: the halo of the tables (non-empty for XLA)
    junk = B + H + dg.n0
    M_l, E_l, I_l = [int(x) for x in M[1:]], [int(x) for x in E[1:]], [int(x) for x in I[1:]]
    cls = [int(np.ceil(np.log2(max(m, 1)))) for m in M_l] if group_by == "log2" else [0] * len(M_l)
    group_sizes = []
    for i, c in enumerate(cls):
        if i > 0 and c == cls[i - 1]:
            group_sizes[-1] += 1
        else:
            group_sizes.append(1)
    groups = []  # (first level, size, M, E, I)
    start = 1
    for size in group_sizes:
        sl = slice(start - 1, start - 1 + size)
        groups.append((start, size, max(max(M_l[sl]), 1), max(max(E_l[sl]), 1), max(max(I_l[sl]), 1)))  # XLA needs non-empty outputs
        start += size

    def fill(nbr_user, init_user, depth, tree_pos, pts):
        nbr, init, depth, tree_pos, pts = nbr_user[0], init_user[0], depth[0], tree_pos[0], pts[0]
        sidx = lax.axis_index(ax)
        h2, halo_depth, exp_depth, halo_rank, ext_idx = common(nbr, init, depth, tree_pos, sidx, H)

        def level_tables(l, Mg, Eg, Ig):
            """Tables of one level at the group's widths."""
            own = jnp.nonzero(depth == l, size=Mg, fill_value=junk)[0]
            ok = own < B
            safe = jnp.where(ok, own, 0)
            ex = jnp.nonzero(exp_depth == l, size=Eg, fill_value=-1)[0]
            im = jnp.nonzero(h2.halo_valid & (halo_depth == l), size=Ig, fill_value=-1)[0]
            ims = jnp.maximum(im, 0)
            return (
                own, jnp.where(ok[:, None], ext_idx[safe], junk), jnp.where(ok[:, None], pts[safe], 0.0), ok.astype(jnp.float64),
                jnp.where(ex >= 0, h2.exp_loc[jnp.maximum(ex, 0)], junk),
                jnp.where(im >= 0, h2.halo[ims] // B, 0), jnp.where(im >= 0, halo_rank[ims], 0), jnp.where(im >= 0, B + ims, junk),
            )

        own_g, nbr_g, pts_g, valid_g, exp_g, src_g, pos_g, slot_g, lvl_g = ([] for _ in range(9))
        for first, size, Mg, Eg, Ig in groups:
            # a few levels at a time: the level masks are (batch, B)
            tabs = lax.map(partial(level_tables, Mg=Mg, Eg=Eg, Ig=Ig), jnp.arange(first, first + size, dtype=jnp.int32), batch_size=min(size, 8))
            for lst, t in zip((own_g, nbr_g, pts_g, valid_g, exp_g, src_g, pos_g, slot_g), tabs):
                lst.append(t[None])
            lvl_g.append(jnp.arange(first, first + size, dtype=jnp.int32)[None])
        init_own = jnp.argsort(jnp.where(init, tree_pos, B))[:n0_s]  # block positions of tree positions 0..n0_s-1
        init_pts = lax.all_gather(pts[init_own], ax, axis=0, tiled=True)  # (n0, d), replicated
        ext_points = jnp.concatenate([pts, h2.gather(pts, 0.0), init_pts, jnp.zeros((1, d), pts.dtype)])
        ext_level = jnp.concatenate([depth, jnp.where(h2.halo_valid, halo_depth, L), jnp.zeros(dg.n0, jnp.int32), jnp.full((1,), L, jnp.int32)])
        init_slot_own = sidx * n0_s + jnp.arange(n0_s)
        return (
            tuple(own_g), tuple(nbr_g), tuple(pts_g), tuple(valid_g), tuple(exp_g), tuple(src_g), tuple(pos_g), tuple(slot_g), tuple(lvl_g),
            ext_points[None], ext_level[None], init_own[None], init_slot_own[None], init_pts, h2.halo[None],
        )

    n_g = len(groups)
    out_specs = ((P(ax),) * n_g,) * 9 + (P(ax),) * 4 + (P(),) + (P(ax),)
    out = smap(fill, in_specs=(P(ax),) * 5, out_specs=out_specs)(dg.neighbors, dg.initial, dg.depth, dg.tree_pos, pts_all)
    jax.block_until_ready(out)
    if verbose:
        logger.info(f"sharded graph tables: {L} levels in {len(groups)} groups, halo {H}, widths up to {max(M_l)}/{max(E_l)}/{max(I_l)} ({time.time() - t0:.1f} s)")
    halo = jnp.where(out[14] < N, out[14], -1)
    init_src = np.arange(dg.n0) // n0_s
    init_pos = np.arange(dg.n0) % n0_s
    return ShardedGraph(S, B, H, dg.n0, L, tuple(group_sizes), *out[:9], out[9], out[10], out[11], out[12], init_src, init_pos, out[13], halo)


# ---------------------------------------------------------------------------- the public device build
def build_sharded_graph(points, mesh: Mesh, *, n0: int = 512, k: int = 16, cuda: bool = True, verbose: bool = False) -> ShardedGraph:
    """Build the graph of `points` (``(..., dim)``, user order, split into
    contiguous blocks along the leading axis over the one-dimensional `mesh`)
    on the devices and assemble its :class:`ShardedGraph` tables; see the
    module docstring. `n0` is the global number of initial points (``n0 //
    n_shards`` per block, at least `k`)."""
    dg = _build_device_graph(points, mesh, n0=n0, k=k, cuda=cuda, verbose=verbose)
    return _compact_tables(dg, mesh, verbose=verbose)


# ---------------------------------------------------------------------------- generation on the mesh
class _ShardedGenerate:
    """Generation of a graphgp GP from a :class:`ShardedGraph` on a mesh
    (``xi`` block-sharded along its leading axis, covariance table replicated);
    the custom derivatives of the module docstring. Used by
    :class:`~nifty.re.gaussian_process.GraphGP`."""

    def __init__(self, graph: ShardedGraph, mesh: Mesh, *, cuda: bool, radial_bins):
        self.g, self.mesh, self.cuda = graph, mesh, bool(cuda)
        self.axis = mesh.axis_names[0]
        self.radial_bins = radial_bins

    @staticmethod
    def _factors(cov, nbr_pts, pts, valid):
        """Mean weights (M, k) and std (M,) of the level's points given their
        neighbours (graphgp's math); padded points get the identity."""
        import graphgp as ggp

        k = nbr_pts.shape[1]
        joint = jnp.concatenate([nbr_pts, pts[:, None, :]], axis=1)
        K = jax.vmap(ggp.compute_cov_matrix, in_axes=(None, 0, 0))(cov, joint, joint)
        K = jnp.where((valid > 0)[:, None, None], K, jnp.eye(k + 1, dtype=K.dtype))
        Lc = jnp.linalg.cholesky(K)
        mean_vec = jnp.linalg.solve(Lc[:, :k, :k].transpose(0, 2, 1), Lc[:, k, :k][..., None]).squeeze(-1)
        return mean_vec, Lc[:, k, k]

    def _update(self, ext, xi, cov_vals, ext_points, tabs):
        """New values (M,) of the level's points from the extended array `ext`
        (the neighbours' coordinates and values are gathered first, so that the
        kernels only see M x k arrays)."""
        own, nbr, pts, valid, exp, src, pos, slot, lvl = tabs
        nbr_pts = ext_points[nbr]
        nbr_vals = ext[nbr]
        if self.cuda:
            import graphgp_cuda

            M, k = nbr.shape
            lv = jnp.where(valid > 0, 1, 0).astype(jnp.int32)
            # behind a barrier: XLA otherwise folds the index pattern into a constant per level group
            flat = lax.optimization_barrier(jnp.arange(M * k, dtype=jnp.int32)).reshape(M, k)
            return graphgp_cuda.refine_level(
                nbr_pts.reshape(M * k, -1), pts, flat, lv, jnp.ones((1,), jnp.int32), self.radial_bins, cov_vals,
                nbr_vals.reshape(M * k), xi[own], ext[own],
            )
        mean_vec, std = self._factors((self.radial_bins, cov_vals), nbr_pts, pts, valid)
        return jnp.sum(mean_vec * nbr_vals, axis=1) + std * xi[own]

    @staticmethod
    def _set(ext, new, tabs):
        own, valid = tabs[0], tabs[3]
        return ext.at[own].set(jnp.where(valid > 0, new, ext[own]))

    def _exchange(self, ext, tabs):
        """Halo exchange after a level: export the own values other shards need,
        one all_gather, fill the halo slots."""
        exp, src, pos, slot = tabs[4:8]
        gathered = lax.all_gather(ext[exp], self.axis, axis=0, tiled=False)  # (S, E)
        return ext.at[slot].set(gathered[src, pos])

    def _exchange_transpose(self, ct_ext, tabs, junk):
        """Transpose of :meth:`_exchange`: the halo slots' cotangents go back to
        the exporting shards (psum_scatter, the transpose of the all_gather)."""
        exp, src, pos, slot = tabs[4:8]
        n_shards = self.mesh.shape[self.axis]
        ct_slots = jnp.where(slot != junk, ct_ext[slot], 0.0)
        ct_gathered = jnp.zeros((n_shards, exp.shape[0]), ct_ext.dtype).at[src, pos].add(ct_slots)
        ct_ext = ct_ext.at[slot].set(0.0)
        ct_buf = lax.psum_scatter(ct_gathered, self.axis, scatter_dimension=0, tiled=False)  # (E,)
        return ct_ext.at[exp].add(ct_buf)

    def _initial(self, xi, cov_vals, g: ShardedGraph):
        """Dense generation of the initial set (replicated): each shard exports the
        excitations of its own initial points, one all_gather."""
        import graphgp as ggp

        gathered = lax.all_gather(xi[g.init_own[0]], self.axis, axis=0, tiled=False)
        xi0 = gathered[g.init_src, g.init_pos]
        return ggp.generate_dense(g.initial_points, (self.radial_bins, cov_vals), xi0)

    @staticmethod
    def _place_initial(ext, initial, g: ShardedGraph):
        B, H, n0 = g.block_size, g.halo_size, g.n0
        ext = ext.at[B + H : B + H + n0].set(initial)
        return ext.at[g.init_own[0]].set(initial[g.init_slot_own[0]])

    @staticmethod
    def _tables(g: ShardedGraph, grp):
        return tuple(t[grp][0] for t in (g.own, g.nbr, g.pts, g.valid, g.exp, g.imp_src, g.imp_pos, g.imp_slot, g.lvl))

    @staticmethod
    def _pad_xi(xi_block, g: ShardedGraph):
        xi = xi_block.reshape(-1)
        return jnp.concatenate([xi, jnp.zeros((g.halo_size + g.n0 + 1,), xi.dtype)])  # junk reads give 0

    # ---- forward
    def _fwd_local(self, xi_block, cov_vals, g: ShardedGraph):
        """The block's values and the final extended array (block, halo, initial)."""
        xi = self._pad_xi(xi_block, g)
        ext_points = g.ext_points[0]
        ext = jnp.zeros((g.junk + 1,), xi.dtype)
        ext = self._place_initial(ext, self._initial(xi, cov_vals, g), g)
        for grp in range(g.n_groups):

            def step(ext, tabs):
                ext = self._set(ext, self._update(ext, xi, cov_vals, ext_points, tabs), tabs)
                return self._exchange(ext, tabs), None

            ext = lax.scan(step, ext, self._tables(g, grp))[0]
        values = ext[: g.block_size]
        values = jnp.where(jnp.any(jnp.isnan(values)), jnp.full_like(values, jnp.nan), values)
        return values, ext

    # ---- derivatives: level by level from the final extended array, nothing stored per level.
    # A value is final once generated, so the input of level l is the final extended array masked
    # to the entries generated before l (their generation level is in `ext_level`, halo included).
    def _tangent_local(self, d_xi_block, d_cov, ext_final, xi_block, cov_vals, g: ShardedGraph):
        xi, d_xi = self._pad_xi(xi_block, g), self._pad_xi(d_xi_block, g)
        ext_points, ext_level = g.ext_points[0], g.ext_level[0]
        _, d_initial = jax.jvp(lambda x, c: self._initial(x, c, g), (xi, cov_vals), (d_xi, d_cov))
        d_ext = self._place_initial(jnp.zeros((g.junk + 1,), xi.dtype), d_initial, g)
        for grp in range(g.n_groups):

            def step(d_ext, tabs):
                lvl = tabs[-1]
                mask = ext_level < lvl
                before = jnp.where(mask, ext_final, 0.0)
                d_before = jnp.where(mask, d_ext, 0.0)
                f = lambda b, x, c: self._update(b, x, c, ext_points, tabs)  # noqa: E731
                _, d_new = jax.jvp(f, (before, xi, cov_vals), (d_before, d_xi, d_cov))
                return self._exchange(self._set(d_ext, d_new, tabs), tabs), None

            d_ext = lax.scan(step, d_ext, self._tables(g, grp))[0]
        return d_ext[: g.block_size]

    def _bwd_local(self, ct_block, ext_final, xi_block, cov_vals, g: ShardedGraph):
        """Cotangents of the block's excitations and of the covariance table (summed over shards)."""
        xi = self._pad_xi(xi_block, g)
        ext_points, ext_level = g.ext_points[0], g.ext_level[0]
        junk = g.junk
        ct_ext = jnp.zeros((junk + 1,), xi.dtype).at[: g.block_size].set(ct_block.reshape(-1))
        ct_xi = jnp.zeros_like(xi)
        ct_cov = jnp.zeros_like(cov_vals)
        for grp in reversed(range(g.n_groups)):

            def step(carry, tabs):
                ct_ext, ct_xi, ct_cov = carry
                lvl, valid, own = tabs[-1], tabs[3], tabs[0]
                ct_ext = self._exchange_transpose(ct_ext, tabs, junk)
                before = jnp.where(ext_level < lvl, ext_final, 0.0)
                ct_new = jnp.where(valid > 0, ct_ext[own], 0.0)
                ct_ext = ct_ext.at[own].set(jnp.where(valid > 0, 0.0, ct_ext[own]))  # consumed by the update
                f = lambda b, x, c: self._update(b, x, c, ext_points, tabs)  # noqa: E731
                _, vjp = jax.vjp(f, before, xi, cov_vals)
                ct_b, ct_x, ct_c = vjp(ct_new)
                return (ct_ext + ct_b, ct_xi + ct_x, ct_cov + ct_c), None

            (ct_ext, ct_xi, ct_cov), _ = lax.scan(step, (ct_ext, ct_xi, ct_cov), self._tables(g, grp), reverse=True)
        # the initial set: replicated values, their cotangent is the sum of the copies
        B, H, n0 = g.block_size, g.halo_size, g.n0
        init_own, init_slot_own = g.init_own[0], g.init_slot_own[0]
        ct_initial = ct_ext[B + H : B + H + n0]
        ct_initial = ct_initial.at[init_slot_own].add(jnp.where(init_own != junk, ct_ext[init_own], 0.0))
        _, vjp0 = jax.vjp(lambda x, c: self._initial(x, c, g), xi, cov_vals)
        ct_x0, ct_c0 = vjp0(ct_initial)
        ct_xi = ct_xi + ct_x0
        ct_cov = lax.psum(ct_cov + ct_c0, self.axis)
        return ct_xi[:B], ct_cov

    def __call__(self, xi, cov_vals):
        """Field ``(n_points,)`` on the mesh from the block-sharded excitations
        ``xi`` ``(n_points,)`` and the covariance table. Forward and reverse
        mode (and the transpose of either) go through the level-by-level
        tangent and its transpose, which rebuild every level's input from the
        final extended array."""
        ax = self.axis
        spec = jax.tree_util.tree_map(lambda _: P(ax), self.g)
        spec = ShardedGraph(**{**{f: getattr(spec, f) for f in spec.__dataclass_fields__}, "init_src": P(), "init_pos": P(), "initial_points": P()})
        # the graph tables and the radial bins are explicit arguments of the custom-derivative
        # function and of the shard_map bodies, not closures: when the model is an argument of a
        # jitted function they are tracers, and a custom_jvp closing over tracers fails to lower
        # inside a nested jit ("no constant handler for DynamicJaxprTracer")

        def forward_ext(xi, cov_vals, g, rb):
            f = shard_map(lambda x, c, g, rb: self._with_bins(rb)._fwd_local(x, c, g), mesh=self.mesh, in_specs=(P(ax), P(), spec, P()), out_specs=(P(ax), P(ax)), check_vma=False)
            return f(xi, cov_vals, g, rb)

        def tangent(ext, xi, cov_vals, d_xi, d_cov, g, rb):
            f = shard_map(lambda *a: self._with_bins(a[-1])._tangent_local(*a[:-1]), mesh=self.mesh, in_specs=(P(ax), P(), P(ax), P(ax), P(), spec, P()), out_specs=P(ax), check_vma=False)
            return f(d_xi, d_cov, ext, xi, cov_vals, g, rb)

        def cotangent(ext, xi, cov_vals, ct, g, rb):
            b = shard_map(lambda *a: self._with_bins(a[-1])._bwd_local(*a[:-1]), mesh=self.mesh, in_specs=(P(ax), P(ax), P(ax), P(), spec, P()), out_specs=(P(ax), P()), check_vma=False)
            return b(ct, ext, xi, cov_vals, g, rb)

        @jax.custom_jvp
        def generate(xi, cov_vals, g, rb):
            return forward_ext(xi, cov_vals, g, rb)[0]

        @generate.defjvp
        def generate_jvp(primals, tangents):
            xi, cov_vals, g, rb = primals
            d_xi, d_cov = tangents[:2]  # the tables and bins are constants of the model
            values, ext = forward_ext(xi, cov_vals, g, rb)
            # the linear tangent map and its transpose as a custom_linear_solve with the identity as
            # matrix: jit, vmap, linear_transpose and vjp all work through it
            rhs = {"xi": d_xi, "cov": d_cov}

            def solve(_, b):
                return {"xi": tangent(ext, xi, cov_vals, b["xi"], b["cov"], g, rb), "cov": jnp.zeros_like(b["cov"])}

            def transpose_solve(_, ct):
                ct_xi, ct_cov = cotangent(ext, xi, cov_vals, ct["xi"], g, rb)
                return {"xi": ct_xi, "cov": ct_cov}

            d_values = lax.custom_linear_solve(lambda x: x, rhs, solve, transpose_solve=transpose_solve)["xi"]
            return values, d_values

        return generate(xi, cov_vals, self.g, self.radial_bins)

    def _with_bins(self, radial_bins):
        """A copy using `radial_bins` (the shard_map bodies' local view of the bins)."""
        other = copy.copy(self)
        other.radial_bins = radial_bins
        return other
