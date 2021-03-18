from jax import numpy as np
import sys
import time


def NCG(pos, energy_vag, met, *args, compiles=False, **kwargs):
    from .field import Field
    from .sugar import makeField
    if isinstance(pos, Field):
        return field_NCG(pos, energy_vag, met, *args, **kwargs)

    def e_vag(field_x):
        x = field_x.to_tree()
        v, g = energy_vag(x)
        return v, makeField(g)

    def m(field_x, field_tan):
        x = field_x.to_tree()
        tan = field_tan.to_tree()
        return makeField(met(x, tan))

    fld = makeField(pos)
    return field_NCG(fld, e_vag, m, *args, **kwargs).to_tree()


def field_NCG(
    pos,
    energy_vag,
    met,
    iterations,
    absdelta=1.,
    name=None,
    time_threshold=None
):
    energy_diff = 0.
    energy, g = energy_vag(pos)
    if np.isnan(energy):
        raise ValueError("energy is Nan")
    for i in range(iterations):
        cg_name = name + "CG" if name is not None else None
        nat_g, _ = cg(
            lambda x: met(pos, x),
            g,
            absdelta=absdelta / 100,
            resnorm=g.norm(ord=1) / 2,
            norm_ord=1,
            name=cg_name,
            time_threshold=time_threshold
        )
        dd = nat_g
        new_pos = pos - dd
        new_energy, new_g = energy_vag(new_pos)
        for j in range(6):
            if new_energy <= energy:
                break
            dd = dd / 2
            new_pos = pos - dd
            new_energy, new_g = energy_vag(new_pos)
            if j == 3:
                if name is not None:
                    msg = f"{name}: long line search, resetting"
                    print(msg, file=sys.stderr)
                gam = float(g.squared_norm())
                curv = float(g.dot(met(pos, g)))
                dd = -gam / curv * g
        else:
            new_pos = pos - nat_g
            new_energy, new_g = energy_vag(new_pos)
            print("Warning: Energy increased", file=sys.stderr)
        energy_diff = energy - new_energy
        if name is not None:
            msg = f"{name}: Iteration {i+1} Energy {new_energy:.6e} diff {energy_diff:.6e}"
            print(msg, file=sys.stderr)
        if energy_diff < absdelta and j < 2:
            return new_pos
        energy = new_energy
        pos = new_pos
        g = new_g
        if time_threshold is not None:
            if time.time() > time_threshold:
                break
    return pos


def cg(mat, j, x0=None, *args, **kwargs):
    from .field import Field
    from .sugar import makeField

    if isinstance(j, Field):
        return field_cg(mat, j, x0, *args, **kwargs)

    def m_f(field_x):
        x = field_x.to_tree()
        return makeField(mat(x))

    j_f = makeField(j)
    x0_f = makeField(x0) if x0 is not None else x0
    x, info = field_cg(m_f, j_f, x0_f, *args, **kwargs)
    return x.to_tree(), info


N_RESET = 20


# Taken from nifty
def field_cg(
    mat,
    j,
    x0=None,
    maxiter=None,
    absdelta=1.,
    resnorm=None,
    norm_ord=2,
    name=None,
    time_threshold=None,
    miniter=5
):
    maxiter = 200 if maxiter is None else maxiter

    if x0 is None:
        pos = 0. * j
        r = -j
        d = r
        # energy = .5xT M x - xT j
        energy = 0.
    else:
        pos = x0
        r = mat(pos) - j
        d = r
        energy = float(((r - j) / 2).dot(pos))
    previous_gamma = float(r.squared_norm())
    if previous_gamma == 0:
        info = 0
        return pos, info

    info = -1
    # TODO(Gordian): Use `lax.while_loop`?
    for i in range(maxiter):
        if name is not None:
            print(f"{name}: Iteration {i} Energy {energy:.6e}", file=sys.stderr)
        q = mat(d)
        curv = float(d.dot(q))
        if curv == 0.:
            raise ValueError("zero curvature in conjugate gradient")
        alpha = previous_gamma / curv
        if alpha < 0:
            raise ValueError("implausible gradient scaling `alpha < 0`")
        pos = pos - alpha * d
        if i % N_RESET == N_RESET - 1:
            r = mat(pos) - j
        else:
            r = r - q * alpha
        gamma = float(r.squared_norm())
        if time_threshold is not None:
            if time.time() > time_threshold:
                info = i
                return pos, info
        if gamma == 0:
            nm = "CG" if name is None else name
            print(f"{nm}: gamma=0, converged!", file=sys.stderr)
            info = 0
            return pos, info
        if resnorm is not None:
            norm = float(r.norm(ord=norm_ord))
            if name is not None:
                msg = f"{name}: gradnorm {norm:.6e} tgt {resnorm:.6e}"
                print(msg, file=sys.stderr)
            if norm < resnorm and i > miniter:
                info = 0
                return pos, info
        new_energy = float(((r - j) / 2).dot(pos))
        if absdelta is not None:
            if name is not None:
                msg = f"{name}: ΔEnergy {energy-new_energy:.6e} tgt {absdelta:.6e}"
                print(msg, file=sys.stderr)
            if energy - new_energy < absdelta and i > miniter:
                info = 0
                return pos, info
        energy = new_energy
        d = d * max(0, gamma / previous_gamma) + r
        previous_gamma = gamma
    else:
        nm = "CG" if name is None else name
        print(f"{nm}: Iteration Limit Reached", file=sys.stderr)
        info = i
    return pos, info
