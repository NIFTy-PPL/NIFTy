import nifty6 as ift
import numpy as np

def optimize_node(op):
    """Takes an operator node (_OpSum or _OpProd) and searches for similar substructures in both vertices.
    Only searches one vertex deep.
    """
    if isinstance(op, ift.operators.operator._OpSum):
        sum = True
    elif isinstance(op, ift.operators.operator._OpProd):
        sum = False
    else:
        return op

    def to_list(x):
        if isinstance(x, ift.operators.operator._OpChain):
            op_list = list(reversed(x._ops))
        else:
            op_list = [x,]
        return op_list

    op_list = [to_list(op._op1), to_list(op._op2)]
    first_difference=0
    try:
        while op_list[0][first_difference] is op_list[1][first_difference]:
            first_difference = first_difference + 1
    except IndexError:
        pass
    if first_difference == 0:
        return op

    common_op = op_list[0][:first_difference]
    res_op = common_op[-1]
    for ops in reversed(common_op[:-1]):
        res_op = res_op @ ops

    performance_adapter = ift.FieldAdapter(res_op.target, str(id(res_op)))

    vertex = [0, 0]
    for i in range(len(op_list)):
        op_list[i][:first_difference] = [performance_adapter]
        vertex[i] = op_list[i][-1]
        for ops in reversed(op_list[i][:-1]):
            vertex[i] = vertex[i] @ ops

    # This seems broken
    # op._op1 = vertex[0]
    # op._op2 = vertex[1]
    # op._domain = ift.sugar.domain_union((op._op1.domain, op._op2.domain))
    if sum: op = vertex[0] + vertex[1]
    else: op = vertex[0] * vertex[1]
    op = op.partial_insert(performance_adapter.adjoint(res_op))
    return op

def optimize_all_nodes(op):
    """Traverses the tree and applies optimization to every node"""
    if isinstance(op, ift.operators.operator._OpChain):
        x = op._ops[-1]
        chained = True
    else:
        x = op
        chained = False
    if isinstance(x, ift.operators.operator._OpSum) or isinstance(x, ift.operators.operator._OpProd):
        #postorder traversing
        x._op1 = optimize_all_nodes(x._op1)
        x._op2 = optimize_all_nodes(x._op2)
        x = optimize_node(x)

    if chained:
        op._ops = op._ops[:-1] + (x,)
        op._domain = x.domain
    else:
        op = x
    return op

#
# Some Examples for above
#

class CountingOp(ift.LinearOperator):
    def __init__(self, domain):

        self._domain = self._target = ift.sugar.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._count = 0

    def apply(self, x, mode):
        self._count += 1
        return x

    @property
    def count(self):
        return self._count

dom = ift.DomainTuple.scalar_domain()
h = ift.from_random('normal', dom)

copuni = CountingOp(dom)
copig = CountingOp(dom)
uni = copuni @ ift.UniformOperator(dom)
ig = copig @ ift.InverseGammaOperator(dom, 1, 1)

op = uni + ig(uni)
op(h)
print(copuni.count) # is increased by 2
print(copig.count)

op2 = optimize_all_nodes(op)
op2(h)
print(copuni.count) # only increased by 1
print(copig.count)

np.allclose(op(h).val, op2(h).val)

# More complex things work partially:
op = ig(uni + uni +uni(ig))
op2 = optimize_all_nodes(op)
# However, since the search depth is only one vertex, this is not optimised:
op = ig(uni + uni(ig) + uni)
op2 = optimize_all_nodes(op)


# To find bigger chunks of subtrees the following function is defined:

def optimize_subtrees(op):
    """Recognizes same subtrees and replaces them.
    Currently only works on operators defined on Multidomains.
    Should work inplace"""
    if not isinstance(op.domain, ift.MultiDomain):
        raise TypeError('Operator needs to be defined on a multidomain')
    dic = {}
    dic_list = {}
    def equal_vertices(x, coord):
        if isinstance(x, ift.operators.operator._OpChain):
            try:
                # Might be better to save parent + left or right
                # However, this makes adjusting the operator domains hard later on
                dic[id(x)] = [coord, ] + dic[id(x)]
                #dic_list[id(x)] =  dic_list[id(x)]
            except KeyError:
                dic[id(x)] = [coord, ]
                dic_list[id(x)] = x
            x = x._ops[-1]

        if isinstance(x, ift.operators.operator._OpSum) or isinstance(x, ift.operators.operator._OpProd):
            equal_vertices(x._op1, coord + [1])
            equal_vertices(x._op2, coord + [2])

    equal_vertices(op, [])

    key = None
    #Heuristically, the first entry should be the largest subtree
    for items in list(dic.items()):
        if len(items[1]) > 1:
            # Multiple Ops are the same
            key = items[0]
            break

    if key is None:
        return op

    same_op = dic_list[key]
    performance_adapter = ift.FieldAdapter(same_op.target, str(key))

    visited = []
    for coord_list in dic[key]:
        x = op
        for coord in coord_list[:-1]:
            # Travel to the nodes
            if isinstance(x, ift.operators.operator._OpChain):
                visited.append(x)
                x = x._ops[-1]
            visited.append(x)
            if coord == 1:
                x = x._op1
            else:
                x = x._op2
        if isinstance(x, ift.operators.operator._OpChain):
            visited.append(x)
            x = x._ops[-1]
        visited.append(x)
        # Substitute subtrees
        if coord_list[-1] == 1:
            x._op1 = performance_adapter
            x._op1._domain = performance_adapter.domain
        else:
            x._op2 = performance_adapter
            x._op2._domain = performance_adapter.domain
        for v in reversed(visited):
            if isinstance(v, ift.operators.operator._OpChain):
                v._domain = v._ops[-1].domain
            if isinstance(v, ift.operators.operator._OpSum) or isinstance(v, ift.operators.operator._OpProd):
                v._domain = ift.sugar.domain_union((v._op1.domain, v._op2.domain))

    op = op.partial_insert(performance_adapter.adjoint(same_op))
    op = optimize_subtrees(op)
    return op

# Examples for operator above

dom = ift.UnstructuredDomain([10000])

uni = ift.UniformOperator(dom)
# It needs to be defined on a multidomain, so that one can recognize the leaves
uni_t = uni.ducktape('test')
ig = ift.InverseGammaOperator(dom, 1, 1)


#Now this is optimised:
op = ig(uni_t + ig(uni_t) + uni_t)
h = ift.from_random('normal', op.domain)
# %timeit op(h)
# 2.7 ms
op = optimize_all_nodes(op)
# %timeit op(h)
# 2.3 ms
op = optimize_subtrees(op)
# %timeit op(h)
# 2.3 ms (only replaces one very fast uni operation in this example)

# However, still some improvements:
# 0. Bugs

# 1. Currently only searching for nodes at the end of operator chains, but
optimize_all_nodes( (uni + uni)(uni + uni) )
#    can happen

# 2. Subtrees are only replaced at node points, should also compare the vertices above and replace it

# 3. Only comparing by ids is done, one might add a cache to prevent multiple operators from going unnoticed,
#    even though this shouldn't be the norm
optimize_subtrees(op = ig(uni_t + ig(uni_t) + uni_t))
optimize_subtrees(op = ig(uni.ducktape('test') + ig(uni.ducktape('test')) + uni.ducktape('test')))

