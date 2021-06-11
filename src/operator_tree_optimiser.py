# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from copy import deepcopy

from numpy import allclose

from .multi_field import MultiField
from .operators.operator import _OpChain, _OpProd, _OpSum
from .operators.simple_linear_operators import FieldAdapter
from .sugar import domain_union, from_random
from .utilities import myassert


def _optimise_operator(op):
    """
    optimises operator trees, so that same operator subtrees are not computed twice.
    Recognizes same subtrees and replaces them at nodes.
    Recognizes same leaves and structures them.
    Works partly inplace, rendering the old operator unusable"""

    # Format: List of tuple [op, parent_index, left=True right=False]
    nodes = []
    # Format: ID: index in nodes[]
    id_dic = {}
    # Format: [parent_index, left]
    leaves = set()

    # helper functions
    def readable_id():
        # Gives out letters to prepend field_adapter ids for cosmetics
        # Start at 'A'
        current_letter = 65
        repeats = 1
        while True:
            yield chr(current_letter)*repeats
            current_letter += 1
            if current_letter == 91:
                # skip specials
                current_letter += 6
            elif current_letter == 123:
                # End at z and start at AA
                current_letter = 65
                repeats += 1

    prepend_id = readable_id()

    def isnode(op):
        return isinstance(op, (_OpSum, _OpProd))


    def left_parser(left_bool):
        return '_op1' if left_bool else '_op2'

    def get_duplicate_keys(k_list, dic):
        for item in list(dic.items()):
            if len(item[1]) > 1:
                k_list.append(item[0])

    # Main algorithm functions
    def rebuild_domains(index):
        """Goes bottom up to fix domains which were destroyed by plugging in field adapters"""
        cond = True
        while cond:
            op = nodes[index][0]
            for attr in ('_op1', '_op2'):
                if isinstance(getattr(op, attr), _OpChain):
                    getattr(op, attr)._domain = getattr(op, attr)._ops[-1].domain
            if isnode(op):
                # Some problems doing this on non-multidomains, because one side becomes a multidomain and the other not
                try:
                    op._domain = domain_union((op._op1.domain, op._op2.domain))
                except AttributeError:
                    import warnings
                    warnings.warn('Operator should be defined on a MultiDomain')
                    pass

            index = nodes[index][1]
            cond = type(index) is int


    def recognize_nodes(op, active_node, left):
        # If nothing added - is a leaf!
        isleaf = True
        if isinstance(op, _OpChain):
           for i in range(len(op._ops)):
                if isnode(op._ops[i]):
                    nodes.append((op._ops[i], active_node, left))
                    isleaf = False
        elif isnode(op):
            nodes.append((op, active_node, left))
            isleaf = False
        if isleaf:
            leaves.add((active_node, left))


    def equal_nodes(op):
        # BFS-Algorithm which fills the nodes list and id_dic dictionary
        # Does not scan equal subtrees multiple times
        list_index_traversed = 0
        recognize_nodes(op, None, None)

        while list_index_traversed < len(nodes):
            # Visit node
            active_node = nodes[list_index_traversed][0]

            # Check whether exists already
            try:
                id_dic[id(active_node)] = id_dic[id(active_node)] + [list_index_traversed]
                match = True
            except KeyError:
                id_dic[id(active_node)] = [list_index_traversed]
                match = False
            # Check vertices for nodes
            if not match:
                recognize_nodes(active_node._op1, list_index_traversed, True)
                recognize_nodes(active_node._op2, list_index_traversed, False)

            list_index_traversed += 1

    edited = set()

    def equal_leaves(leaves):
        id_leaf = {}
        # Find matching leaves
        def write_to_dic(leaf, leaf_op_id):
            try:
                id_leaf[leaf_op_id] = id_leaf[leaf_op_id] + (leaf,)
            except KeyError:
                id_leaf[leaf_op_id] = (leaf,)

        for leaf in leaves:
            parent = nodes[leaf[0]][0]
            attr = left_parser(leaf[1])
            leaf_op = getattr(parent, attr)
            if isinstance(leaf_op, _OpChain):
                leaf_op_id = ''
                for i in reversed(leaf_op._ops):
                    leaf_op_id += str(id(i))
                    if not isinstance(i, FieldAdapter):
                        # Do not optimise leaves which only have equal FieldAdapters
                        write_to_dic(leaf, leaf_op_id)
                        break
            else:
                if not isinstance(leaf_op, FieldAdapter):
                    write_to_dic(leaf, str(id(leaf_op)))


        # Unroll their OpChain and see how far they are equal
        key_list_leaf = []
        same_leaf = {}
        get_duplicate_keys(key_list_leaf, id_leaf)

        for key in key_list_leaf:
            to_compare = []
            for leaf in id_leaf[key]:
                parent = nodes[leaf[0]][0]
                attr = left_parser(leaf[1])
                leaf_op = getattr(parent, attr)
                if isinstance(leaf_op, _OpChain):
                    to_compare.append(tuple(reversed(leaf_op._ops)))
                else:
                    to_compare.append((leaf_op,))
            first_difference = 1
            max_diff = min(len(i) for i in to_compare)
            if not max_diff == 1:
                compare_iterator = iter(to_compare)
                first = next(compare_iterator)
                while all(first[first_difference] == rest[first_difference] for rest in compare_iterator):
                    first_difference += 1
                    if first_difference >= max_diff:
                        break
                    compare_iterator = iter(to_compare)
                    first = next(compare_iterator)

            common_op = to_compare[0][:first_difference]
            res_op = common_op[0]
            for ops in common_op[1:]:
                res_op = ops @ res_op

            same_leaf[key] = [res_op, FieldAdapter(res_op.target, next(prepend_id) + str(id(res_op)))]

            for leaf in id_leaf[key]:
                parent = nodes[leaf[0]][0]
                edited.add(id_dic[id(parent)][0])
                attr = left_parser(leaf[1])
                leaf_op = getattr(parent, attr)
                if isinstance(leaf_op, _OpChain):
                    if first_difference == len(leaf_op._ops):
                        setattr(parent, attr, same_leaf[key][1])
                    else:
                        leaf_op._ops = leaf_op._ops[:-first_difference] + (same_leaf[key][1],)
                else:
                    setattr(parent, attr, same_leaf[key][1])
        return key_list_leaf, same_leaf

    equal_nodes(op)

    key_temp = []
    key_list_op, same_op = equal_leaves(leaves)
    cond = True
    while cond:
        key_temp, same_op_temp = equal_leaves(leaves)
        key_list_op += key_temp
        same_op.update(same_op_temp)
        cond = len(same_op_temp) > 0
    key_temp.clear()

    # Cut subtrees
    key_list_node = []
    key_list_subtrees = []
    same_node = {}
    same_subtrees = {}
    subtree_leaves = set()

    get_duplicate_keys(key_list_node, id_dic)

    for key in key_list_node:
        same_node[key] = [nodes[id_dic[key][0]][0],
                          FieldAdapter(nodes[id_dic[key][0]][0].target, next(prepend_id) + str(key))]

        for node_indices in id_dic[key]:
            edited.add(node_indices)
            parent = nodes[nodes[node_indices][1]][0]
            attr = left_parser(nodes[node_indices][2])
            if isinstance(getattr(parent, attr), _OpChain):
                getattr(parent, attr)._ops = getattr(parent, attr)._ops[:-1] + (same_node[key][1],)
            else:
                setattr(parent, attr, same_node[key][1])
            # Nodes have been replaced - treat replacements now as leaves
            subtree_leaves.add((nodes[node_indices][1], nodes[node_indices][2]))
        cond = True
        while cond:
            key_temp1, same_temp = equal_leaves(subtree_leaves)
            key_temp = key_temp1 + key_temp
            same_subtrees.update(same_temp)
            cond = len(same_temp) > 0
        key_list_subtrees += key_temp + [key, ]
        key_temp.clear()
        subtree_leaves.clear()
    same_subtrees.update(same_node)

    for index in edited:
        rebuild_domains(index)
    if isinstance(op, _OpChain):
        op._domain = op._ops[-1].domain

    # Insert trees before leaves
    for key in key_list_subtrees:
        op = op.partial_insert(same_subtrees[key][1].adjoint(same_subtrees[key][0]))
    for key in reversed(key_list_op):
        op = op.partial_insert(same_op[key][1].adjoint(same_op[key][0]))
    return op





def optimise_operator(op):
    """
    Merges redundant operations in the tree structure of an operator.
    For example it is ensured that for ``f@x + x`` the operator ``x`` is only computed once.
    It is supposed to be used on the whole operator chain before doing minimisation.

    Currently optimises only ``_OpChain``, ``_OpSum`` and ``_OpProd`` and not their linear pendants
    ``ChainOp`` and ``SumOperator``.

    Parameters
    ----------
    op : Operator
        Operator with a tree structure.

    Returns
    -------
    op_optimised : Operator
        Operator with same input/output, but optimised tree structure.

    Notes
    -----
    Operators are compared only by id, so best results are achieved when the following code

    >>> from nifty8 import UniformOperator, DomainTuple
    >>> uni1 = UniformOperator(DomainTuple.scalar_domain()
    >>> uni2 = UniformOperator(DomainTuple.scalar_domain()
    >>> op = (uni1 + uni2)*(uni1 + uni2)

    is replaced by something comparable to

    >>> uni = UniformOperator(DomainTuple.scalar_domain())
    >>> uni_add = uni + uni
    >>> op = uni_add * uni_add

    After optimisation the operator is as fast as

    >>> op = (2*uni)**2
    """
    op_optimised = deepcopy(op)
    op_optimised = _optimise_operator(op_optimised)
    test_field = from_random(op.domain)
    if isinstance(op(test_field), MultiField):
        for key in op(test_field).keys():
            myassert(allclose(op(test_field).val[key], op_optimised(test_field).val[key], 1e-10))
    else:
        myassert(allclose(op(test_field).val, op_optimised(test_field).val, 1e-10))
    return op_optimised
