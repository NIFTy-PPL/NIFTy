from jax import numpy as np
import numpy as onp
import time

# TODO: Use fields to flatten

def my_flatten(ll):
    struct = [l.shape for l in ll]
    res = np.concatenate([l.flatten() for l in ll])
    return res, struct

def my_unflatten(ar, struct):
    ind = 0
    ll = []
    for i in range(len(struct)):
        nn = onp.prod(struct[i])
        ll += [ar[ind:ind+nn].reshape(struct[i])]
        ind += nn
    return ll

def pytree_NCG(pos, Egv, met, *args, compiles=False, **kwargs):
    ar, struct = my_flatten(pos)
    def my_Egv(flat_x):
        x = my_unflatten(flat_x, struct)
        v,g = Egv(x)
        return v, my_flatten(g)[0]

    def my_met(flat_x, flat_tan):
        x = my_unflatten(flat_x, struct)
        tan = my_unflatten(flat_tan, struct)
        return my_flatten(met(x, tan))[0]

    if compiles:
        from jax import jit
        return my_unflatten(NCG(ar, jit(my_Egv), jit(my_met), *args, **kwargs), struct)
    else:
        return my_unflatten(NCG(ar, my_Egv, my_met, *args, **kwargs), struct)


def NCG(pos, Egv, met, iterations, absdelta=1., name=None, time_threshold=None):
    Ediff = 0.
    Eval,  g = Egv(pos)
    if np.isnan(Eval):
        raise ValueError("energy is Nan")
    for i in range(iterations):
        CGname = name+"CG" if name is not None else None
        Dg = cg(lambda x: met(pos,x), g,
                absdelta=absdelta/100,
                resnorm=np.linalg.norm(g, ord=1)/2,
                norm_ord=1,
                name=CGname,
                time_threshold=time_threshold)
        dd = Dg
        npos = pos - dd
        nEval, ng = Egv(npos)
        for j in range(6):
            if nEval <= Eval:
                break
            dd = dd/2 
            npos = pos - dd
            nEval, ng = Egv(npos)
            if j == 3:
                if name is not None:
                    print("{}: long line search, resetting".format(name))
                gam = np.dot(g,g)
                curv = np.dot(g,met(pos,g))
                dd = -gam/curv*g
        else:
            npos = pos - Dg
            nEval, ng = Egv(npos)
            print("Warning: Energy increased")
        Ediff = Eval - nEval
        if name is not None:
            print("{}: Iteration {} Energy {:.6e} diff {:.6e}".format(name, i+1, nEval, Ediff))
        if Ediff < absdelta and j<2:
            return npos
        Eval = nEval
        pos = npos
        g = ng
        if time_threshold is not None:
            if time.time()>time_threshold:
                break
    return pos

nreset = 20
# Taken from nifty
def cg(mat, j, max_iterations=200, absdelta=1., resnorm=None, norm_ord=2, name=None, time_threshold=None, min_iterations=5):
    pos = np.zeros_like(j)
    # energy = .5xT M x - xT j
    r = -j
    d = r
    previous_gamma = np.dot(r, d)
    Eval = 0.
    for i in range(max_iterations):
        if name is not None:
            print("{}: Iteration {} Energy {:.6e}".format(name, i, Eval))
        q = mat(d)
        curv = np.dot(d,q)
        if curv == 0.:
            raise ValueError("CG: zero curvature")
        alpha = previous_gamma/curv
        if alpha < 0:
            raise ValueError("CG: alpha < 0")
        pos = pos-alpha*d
        if i % nreset == nreset-1:
            r = mat(pos)-j
        else:
            r = r - q*alpha
        gamma = np.dot(r,r)
        if time_threshold is not None:
            if time.time()>time_threshold:
                return pos
        if gamma == 0:
            print("gamma=0, converged!")
            return pos
        if resnorm is not None:
            norm = np.linalg.norm(r, ord=norm_ord)
            if name is not None:
                print("gradnorm {:.6e} tgt {:.6e}".format(norm, resnorm))
            if norm < resnorm and i > min_iterations:
                return pos
        new_Eval = np.dot((r-j)/2,pos)
        if absdelta is not None:
            if name is not None:
                print("DeltaEnergy {:.6e} tgt {:.6e}".format(Eval-new_Eval, absdelta))
            if Eval - new_Eval < absdelta and i > min_iterations:
                return pos
        Eval = new_Eval
        d = d*max(0,gamma/previous_gamma)+r
        previous_gamma = gamma
    else:
        print("Iteration Limit Reached")
    return pos
