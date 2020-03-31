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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from nifty6.operators.operator import _OpChain, _OpSum, _OpProd
from nifty6.sugar import domain_union
from nifty6 import FieldAdapter

def optimize_operator(op):
    """
    Optimizes operator trees, so that same operator subtrees are not computed twice.
    Recognizes same subtrees and replaces them at nodes.
    Recognizes same leaves and structures them.
    Works partly inplace, rendering the old operator unusable"""

    # Format: List of tuple [op, parent_index, left=True right=False, pos in op_chain]
    nodes = []
    # Format: ID: index in nodes[]
    id_dic = {}
    # Format: [parent_index, left]
    leaves = set()

    def isnode(op):
        return isinstance(op, _OpSum) or isinstance(op, _OpProd)

    def left_parser(left_bool):
        if left_bool:
            return '_op1'
        return '_op2'

    def rebuild_domains(index):
        """Goes bottom up to fix domains which were destroyed by plugging in field adapters"""
        cond = True
        while cond:
            op = nodes[index][0]
            for attr in ('_op1', '_op2'):
                if isinstance(getattr(op, attr), _OpChain):
                    getattr(op, attr)._domain =  getattr(op, attr)._ops[-1].domain
            if isnode(op):
                op._domain = domain_union((op._op1.domain, op._op2.domain))

            index = nodes[index][1]
            cond = type(index) is int
        return

    def recognize_nodes(op, active_node, left):
        # If nothing added - is a leave!
        isleave = True
        if isinstance(op, _OpChain):
           for i in range(len(op._ops)):
                if isnode(op._ops[i]):
                    nodes.append((op._ops[i], active_node, left, i))
                    isleave = False
        elif isnode(op):
            nodes.append((op, active_node, left, -1))
            isleave = False
        if isleave:
            leaves.add((active_node, left))
        return

    def equal_nodes(op):
        """BFS-Algorithm which fills the nodes list and id_dic dictionary.
        Does not scan equal subtrees multiple times."""
        list_index_traversed = 0
        # if isnode(op):
        #     nodes.append((op, None, None, None))
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
        return

    equal_nodes(op)

    # Cut subtrees
    key_list_tree = []
    same_tree = {}
    edited = set()

    for item in list(id_dic.items()):
        if len(item[1]) > 1:
            key_list_tree.append(item[0])

    for key in key_list_tree:
        same_tree[key] = [nodes[id_dic[key][0]][0],]
        performance_adapter = FieldAdapter(same_tree[key][0].target, str(key))
        same_tree[key] += [performance_adapter]

        for node_indices in id_dic[key]:
            edited.add(node_indices)
            pos = nodes[node_indices][3]
            parent = nodes[nodes[node_indices][1]][0]
            #TODO: cut subtrees can also be regarded as nodes
            #Some kind of order is needed to get the domains right at the end...
            #leaves.add((node_indices, nodes[node_indices][2]))
            attr = left_parser(nodes[node_indices][2])
            if pos is not -1:
                getattr(parent, attr)._ops = getattr(parent, attr)._ops[:pos] + (performance_adapter,) + \
                                                  getattr(parent, attr)._ops[pos+1:]
            else:
                setattr(parent, attr, performance_adapter)


    # Compare leaves
    id_leave = {}
    # Find matching leaves
    for leave in leaves:
        parent = nodes[leave[0]][0]
        attr = left_parser(leave[1])
        leave_op = getattr(parent, attr)
        if isinstance(leave_op, _OpChain):
            leave_op = leave_op._ops[-1]
        try:
            id_leave[id(leave_op)] = id_leave[id(leave_op)] + (leave,)
        except KeyError:
            id_leave[id(leave_op)] = (leave, )


    # Unroll their OpChain and see how far they are equal
    # TODO: Repeat detection, to handle eg. op2(op) + op3(op) + op2(op)
    key_list_op = []
    same_op = {}

    for item in list(id_leave.items()):
        if len(item[1]) > 1:
            key_list_op.append(item[0])
    for key in key_list_op:
        to_compare = []
        for leave in id_leave[key]:
            parent = nodes[leave[0]][0]
            attr = left_parser(leave[1])
            leave_op = getattr(parent, attr)
            if isinstance(leave_op, _OpChain):
                to_compare.append(tuple(reversed(leave_op._ops)))
            else:
                to_compare.append((leave_op,))
        first_difference = 1
        max_diff = min(len(i) for i in to_compare)
        if not max_diff == 1:
            compare_iterator = iter(to_compare)
            first = next(compare_iterator)
            while all(first[first_difference] == rest[first_difference] for rest in compare_iterator):
                first_difference +=1
                if first_difference >= max_diff:
                    break
                compare_iterator = iter(to_compare)
                first = next(compare_iterator)

        # Do not optimise leaves which only have equal FieldAdapters
        if first_difference > int(isinstance(to_compare[0][-1], FieldAdapter)):
            common_op = to_compare[0][:first_difference]
            res_op = common_op[0]
            for ops in common_op[1:]:
                res_op = ops @ res_op

            same_op[key] = [res_op, FieldAdapter(res_op.target, str(id(res_op)))]

            for leave in id_leave[key]:
                parent = nodes[leave[0]][0]
                edited.add(id_dic[id(parent)][0])
                attr = left_parser(leave[1])
                leave_op = getattr(parent, attr)
                if isinstance(leave_op, _OpChain):
                    if first_difference == len(leave_op._ops):
                        setattr(parent, attr, same_op[key][1])
                    else:
                        leave_op._ops = leave_op._ops[:-first_difference] + (same_op[key][1],)
                else:
                    setattr(parent, attr, same_op[key][1])

    for index in edited:
        rebuild_domains(index)
    if isinstance(op, _OpChain):
        op._domain = op._ops[-1].domain

    # Insert trees before leaves
    for key in key_list_tree:
        op = op.partial_insert(same_tree[key][1].adjoint(same_tree[key][0]))
    for key in key_list_op:
        op = op.partial_insert(same_op[key][1].adjoint(same_op[key][0]))
    return op

from copy import deepcopy
from nifty6 import from_random, MultiField
from numpy import allclose
def optimize_operator_safe(op):
    op_optimized = deepcopy(op)
    op_optimized = optimize_operator(op_optimized)
    test_field = from_random('normal', op.domain)

    if isinstance(op(test_field), MultiField):
        for key in op(test_field).keys():
            assert allclose(op(test_field).val[key], op_optimized(test_field).val[key], 1e-10)
    else:
        assert allclose(op(test_field).val, op_optimized(test_field).val, 1e-10)
    return op_optimized
