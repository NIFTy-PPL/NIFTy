# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from jax import lax

_DISABLE_CONTROL_FLOW_PRIM = False


def cond(pred, true_fun, false_fun, operand):
    if _DISABLE_CONTROL_FLOW_PRIM:
        if pred:
            return true_fun(operand)
        else:
            return false_fun(operand)
    else:
        return lax.cond(pred, true_fun, false_fun, operand)


def while_loop(cond_fun, body_fun, init_val):
    if _DISABLE_CONTROL_FLOW_PRIM:
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val
    else:
        return lax.while_loop(cond_fun, body_fun, init_val)


def fori_loop(lower, upper, body_fun, init_val):
    if _DISABLE_CONTROL_FLOW_PRIM:
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fun(i, val)
        return val
    else:
        return lax.fori_loop(lower, upper, body_fun, init_val)
