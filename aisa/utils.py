#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# File      : utils.py
# Creation  : 01 Sept 2020
#
# Copyright (c) 2020 Mauro Faccin <mauro.fccn@gmail.com>
#               https://maurofaccin.github.io
#
# Description : Utility classes and function for base.py
#
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
# =============================================================================

import numpy as np
from scipy import sparse

np.seterr(all="raise")


class Prob():
    """A class to store the probability p and the plogp value."""

    __slots__ = ["__p", "__plogp"]

    def __init__(self, value):
        """Given a float or a Prob, store p and plogp."""
        # if float(value) < 0.0:
        #     raise ValueError('Must be non-negative.')
        self.__p = float(value)
        if isinstance(value, Prob):
            self.__plogp = value.plogp
        else:
            self.__update_plogp()

    def __update_plogp(self):
        if 0.0 < self.__p < np.inf:
            self.__plogp = self.__p * np.log2(self.__p)
        elif np.isclose(self.__p, 0.0, 1e-13) or np.isclose(self.__p, 1.0):
            self.__plogp = 0.0
        else:
            print(self)
            raise ValueError("A probability should be between 0 and 1 not {}")

    @property
    def plogp(self):
        return self.__plogp

    @property
    def p(self):
        return self.__p

    def copy(self):
        return Prob(self)

    def __float__(self):
        return self.__p

    def __iadd__(self, other):
        self.__p += float(other)
        self.__update_plogp()
        return self

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __isub__(self, other):
        self.__p -= float(other)
        self.__update_plogp()
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __imul__(self, other):
        # update p
        oldp = self.__p
        self.__p *= float(other)

        # update plogp
        if isinstance(other, Prob):
            self.__plogp = other.p * self.plogp + oldp * other.plogp
        else:
            self.__update_plogp()
        return self

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __itruediv__(self, other):
        self.__p /= float(other)
        self.__update_plogp()
        return self

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __repr__(self):
        return "{:g} [{:g}]".format(self.__p, self.__plogp)

    def __eq__(self, other):
        # TODO: add approx?
        return self.__p == float(other)

    # set inverse operators
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__


class SparseMat():
    """A sparse matrix with column and row slicing capabilities"""

    __slots__ = ["_dok", "_nn", "_dim", "_norm", "__p_thr"]

    def __init__(self, mat, node_num=None, normalize=False, plength=None):
        """Initiate the matrix

        :mat: scipy sparse matrix or
                list of ((i, j, ...), w) or
                dict (i, j, ...): w
        :node_num: number of nodes
        :normalize: (bool) whether to normalize entries or not
        :plenght: lenght of each path, to use only if len(mat) == 0
        """
        if isinstance(mat, sparse.spmatrix):
            mat = sparse.coo_matrix(mat)
            self._dok = {
                (i, j): Prob(d) for i, j, d in zip(mat.col, mat.row, mat.data)
            }
            if node_num is None:
                self._nn = mat.shape[0]
            self._dim = 2
        elif isinstance(mat, dict):
            self._dok = {tuple(k): Prob(v) for k, v in mat.items()}
            if node_num is None:
                self._nn = np.max([dd for d in self._dok for dd in d]) + 1
            # get the first key of the dict
            if plength is None:
                val = next(iter(self._dok.keys()))
                self._dim = len(val)
            else:
                self._dim = plength
        else:
            self._dok = {tuple(i): Prob(d) for i, d in mat}
            if node_num is None:
                self._nn = np.max([dd for d in self._dok for dd in d]) + 1
            self._dim = len(mat[0][0])

        if node_num is not None:
            self._nn = node_num

        if isinstance(normalize, (float, Prob)):
            self._norm = Prob(normalize)
        elif normalize:
            vsum = np.sum([float(v) for v in self._dok.values()])
            self._norm = Prob(vsum)
        elif not normalize:
            self._norm = Prob(1.0)
        else:
            raise ValueError()

        if self._norm == 0.0 and len(self._dok) > 0:
            raise ValueError("This is a zero matrix")

        self.__update_all_paths()

    def entropy(self):
        """Return the entropy
        (assuming this matrix is a probability distribution)
        """
        if self._nn == 0:
            return 0.0
        sum_plogp = np.sum([p.plogp for p in self._dok.values()])
        return (self._norm.plogp - sum_plogp) / self._norm.p

    @property
    def shape(self):
        """Return the shape of the tensor"""
        return tuple([self._nn] * self._dim)

    @property
    def nn(self):
        return self._nn

    def checkme(self):
        log.info(
            "{} -- NN {}; NL {}".format(
                self.__class__.__name__, self._nn, len(self._dok)
            )
        )

    def size(self):
        return len(self._dok)

    def project(self, part, move_node=None):
        """Returns a new SparseMat projected to part"""
        _part = part.copy()

        # if a node needs to be reassigned
        if move_node is not None:
            # old_part = _part[move_node[0]]
            _part[move_node[0]] = move_node[1]

        new_dok = {}
        for path, val in self._dok.items():
            new_indx = tuple(_part[i] for i in path)

            new_dok.setdefault(new_indx, 0.0)
            new_dok[new_indx] += val.copy()

        return SparseMat(new_dok, node_num=_part.np, normalize=self._norm)

    def copy(self):
        return SparseMat(
            {path[:]: w.copy() for path, w in self._dok.items()},
            node_num=self._nn,
            normalize=self._norm,
            plength=self._dim,
        )

    def dot(self, other, indx):
        if not isinstance(other, np.ndarray):
            raise TypeError(
                "other should be numpy.ndarray, not {}".format(type(other))
            )
        out = np.zeros_like(other, dtype=float)
        for path, w in self._dok.items():
            out[path[indx]] += float(w) * other[path[1 + indx]]
        return out / self._norm.p

    def get_egonet(self, inode, axis=None):
        """Return the adjacency matrix of the ego net of node node."""
        if axis is None:
            slist = [(p, self._dok[p]) for p in self.__p_thr[inode]]
        else:
            slist = [
                (p, self._dok[p])
                for p in self.__p_thr[inode]
                if p[axis] == inode
            ]
        if len(slist) < 1:
            return None
        return SparseMat(slist, node_num=self._nn, normalize=self._norm)

    def get_submat(self, inodes):
        return SparseMat(
            {
                p: self._dok[p]
                for p in set().union(*[self.__p_thr[i] for i in inodes])
            },
            node_num=self._nn,
            normalize=self._norm,
        )

    def slice(self, axis=0, n=0):
        if axis == 0:
            vec = [self._dok.get((n, nn), 0.0) for nn in range(self._nn)]
        else:
            vec = [self._dok.get((nn, n), 0.0) for nn in range(self._nn)]
        return np.array(vec)

    def get_random_entry(self, return_all_probs=False):
        probs = np.array([float(n) for n in self._dok.values()])
        probs /= probs.sum()

        # choose one neighbour based on probs
        link_id = np.random.choice(len(self._dok), p=probs)
        link_prob = probs[link_id]
        link = list(self._dok.keys())[link_id]
        if return_all_probs:
            return link, link_prob, probs
        return link, link_prob

    def paths_through_node(self, node, position=0):
        return [p for p in self.__p_thr[node] if p[position] == node]

    def paths(self, axis=None, node=None):
        if axis is None or node is None:
            yield from self.__iter__()

        else:
            for p, v in self._dok.items():
                if p[axis] == node:
                    yield p, v / self._norm

    def set_path(self, path, weight):
        """ Overwrite path weight. """
        self._dok[path] = Prob(weight) * self._norm
        for i in path:
            self.__p_thr[i].add(path)

    def get_from_sparse(self, other, normalize=False):
        return SparseMat(
            {p: self._dok[p] for p, _ in other if p in self._dok},
            node_num=self._nn,
            normalize=normalize,
        )

    def get_from_paths(self, paths, normalize=False):
        return SparseMat(
            {p: self._dok[p] for p in paths if p in self._dok},
            node_num=self._nn,
            normalize=normalize,
        )

    def add_colrow(self):
        self._nn += 1
        self.__p_thr.append(set())
        return self._nn - 1

    def merge_colrow_bak(self, index_from, index_to):
        """Merge two indexes in each dimension.
        Merge locally.
        """
        if index_from == index_to:
            return

        for path in self.__p_thr[index_from]:
            new_path = tuple(
                [p if p != index_from else index_to for p in path]
            )

            prob = self._dok.pop(path)
            self._dok.setdefault(new_path, 0.0)
            self._dok[new_path] += prob

            self.__p_thr[index_to].add(new_path)

        # forgot to compact indices
        self._nn -= 1

    def merge_colrow(self, index_from, index_to):
        """Merge two indexes in each dimension."""
        if index_from == index_to:
            return self.copy()

        # indx1, indx2 = sorted([index1, index2])
        new_dict = {}
        for path, value in self._dok.items():
            # change partition
            newpath = [p if p != index_from else index_to for p in path]
            # compact indices
            newpath = tuple(p - int(p > index_from) for p in newpath)
            # newpath = tuple(
            #     i - int(i > indx2) if i != indx2 else indx1 for i in path
            # )

            new_dict.setdefault(newpath, 0.0)
            new_dict[newpath] += value

        return SparseMat(new_dict, node_num=self._nn - 1, normalize=self._norm)

    def kron(self, other):
        dok = {}
        for n in range(self._nn):
            for pA in self.paths_through_node(n, position=-1):
                for pB in other.paths_through_node(n, position=0):
                    dok[pA[:-1] + pB] = self._dok[pA] * other._dok[pB]

        return SparseMat(
            dok,
            node_num=self._nn,
            normalize=self._norm * other._norm
            # normalize=True
        )

    def sum(self, axis=None):
        # return the sum of all entries
        if axis is not None:
            probs = np.zeros(self._nn)
            for p, v in self._dok.items():
                probs[p[axis]] += float(v)
            return probs / float(self._norm)
        if self._nn == 0:
            return 0.0
        return np.sum([float(p) for p in self._dok.values()]) / float(
            self._norm
        )

    def __update_all_paths(self):
        """ For each node, all paths that go through it."""
        self.__p_thr = [set() for _ in range(self._nn)]
        for path in self._dok.keys():
            for i in path:
                try:
                    self.__p_thr[i].add(path)
                except IndexError:
                    print(path, self._nn)
                    raise

    def __or__(self, other):
        """ Return a SparseMat with entries from both self and other.
        Local entries will be overwritten by other's.
        """
        new = self.copy()
        for p, v in other:
            new.set_path(p, v)
        return new

    def __iter__(self):
        for k, v in self._dok.items():
            yield k, v / self._norm

    def __getitem__(self, item):
        try:
            return self._dok[item] / self._norm
        except KeyError:
            return 0.0

    def __iadd__(self, other):
        ratio = self._norm / other._norm
        for p, d in other._dok.items():
            if p in self._dok:
                self._dok[p] += d * ratio
            else:
                self._dok[p] = d * ratio
            for i in p:
                self.__p_thr[i].add(p)
        return self

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __isub__(self, other):
        ratio = self._norm / other._norm
        """Can provide negative values."""
        for path, prob in other._dok.items():
            prob_norm = prob * ratio
            lprob = self._dok.get(path, None)
            if lprob is None:
                print('AAA', prob, ratio, path, prob_norm)
                raise ValueError
            if np.isclose(float(lprob), float(prob_norm), atol=1e-12):
                for i in path:
                    self.__p_thr[i].discard(path)
                del self._dok[path]
            else:
                # no need to update __p_thr
                self._dok[path] -= prob_norm
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __imul__(self, other):
        if self._nn != other._nn:
            raise ValueError(
                    "Impossible to multiply matrices of different sizes"
                    f" {self._nn} and {other._nn}"
                )
        self._norm *= other._norm
        if self.size() < other.size():
            keys = [k for k in self._dok if k in other._dok]
        else:
            keys = [k for k in other._dok if k in self._dok]
        new_dok = {k: other._dok[k] * self._dok[k] for k in keys}

        self._dok = new_dok
        self.__update_all_paths()
        return self

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __eq__(self, other):
        for p, v in self._dok.items():
            if not np.isclose(
                float(v / self._norm),
                float(other._dok[p] / other._norm),
                atol=1e-10,
            ):
                return False
        return True


class Partition():
    """a bidirectional dictionary to store partitions. (for internal use)"""

    def __init__(self, partition: dict):
        """get a dictionary node->class"""
        if partition is None:
            self.n2i = None
            self.i2n = None
            self.parts = None
            self.partition = {}
        else:
            self.from_dictionary(partition)

    def node_names(self):
        yield from self.n2i

    def to_dictionary(self):
        """Return a dictionary node->part (usign original names)"""
        return {self.i2n[inode]: part for inode, part in self.items()}

    def from_dictionary(self, partition):
        """Set partition from a node->class dictionary"""

        # save names for later use
        # map names to index
        self.n2i = {n: i for i, n in enumerate(partition.keys())}
        # iverse map
        self.i2n = {i: n for n, i in self.n2i.items()}

        p2i = {p: i for i, p in enumerate(set(partition.values()))}
        self.partition = {
            self.n2i[n]: p2i[p] for n, p in partition.items()
        }
        # self = dict(partition)

        self.parts = [set() for i in p2i.values()]
        for node, part in self.items():
            self.parts[part].add(node)

    def to_coo(self):
        """Return a NxM projection matrix"""
        return partition2coo_sparse(self)

    @property
    def np(self):
        """Return the number of partitions"""
        return len(self.parts)

    def __setitem__(self, node, part):
        """Set i_nodes to partition."""

        old_part = self[node]
        if part == old_part:
            return

        # remove node from old partition
        if len(self.parts[old_part]) == 1:
            if self.np == part:
                # do not move a sigleton to an empty partition
                return
            self.merge(old_part, part)
            return

        # set node partition
        self.partition[node] = part

        # add node to new partition
        if part == len(self.parts):
            self.parts.append(set())
        self.parts[part].add(node)

    def __len__(self):
        return len(self.partition)

    def __delitem__(self, node):
        """Remove nodes."""
        raise AttributeError('Nodes cannot be removed')

    def merge(self, part_from, part_to):
        nodes_to_move = self.parts[part_from].copy()

        for node in nodes_to_move:
            self.partition[node] = part_to

        # compact indices
        for node, part in self.items():
            if part > part_from:
                self.partition[node] = part - 1

        self.parts[part_to] |= self.parts[part_from]
        self.parts.pop(part_from)

    def items(self):
        yield from self.partition.items()

    def __getitem__(self, node):
        return self.partition[node]

    def copy(self):
        return Partition(self.to_dictionary())

    def keys(self):
        yield from self.partition.keys()

    def values(self):
        yield from self.partition.values()


class Bipartition(dict):
    """a bidirectional dictionary to store partitions."""

    def __init__(self, *args, **kwargs):
        """get a dictionary node->class"""
        super(partition, self).__init__(*args, **kwargs)
        self.part = {}
        for key, value in self.items():
            self.part.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.part[self[key]].remove(key)
        super(partition, self).__setitem__(key, value)
        self.part.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.part.setdefault(self[key], []).remove(key)
        if self[key] in self.part and not self.part[self[key]]:
            del self.part[self[key]]
        super(partition, self).__delitem__(key)


def entropy(array):
    """Compute entropy."""
    # if array is a scalar
    if isinstance(array, (float, np.float32, np.float64)):
        if array <= 0.0:
            return 0.0
        return -array * np.log2(array)

    # if it is a vector or matrix
    try:
        # if it store plogp
        return array.entropy()
    except AttributeError:
        # otherwise use numpy
        array = np.array(array)
        array = array[array > 0]
        return -np.sum(array * np.log2(array))


def get_probabilities(
        edges,
        node_num,
        symmetric=False,
        return_transition=False,
        compute_steady=False,
        T=None
        ):
    """Compute p_ij and p_i at the steady state"""

    graph = edgelist2csr_sparse(edges, node_num)
    if symmetric:
        graph += graph.transpose()
    steadystate = graph.sum(0)

    diag = sparse.spdiags(1.0 / steadystate, [0], node_num, node_num)
    transition = graph @ diag

    if T is not None:
        transition = transition ** T

    if compute_steady:
        diff = 1.0
    else:
        # go into the loop only if I need to compute the steady state recursively
        diff = 0.0
    count = 0
    steadystate = np.array(steadystate).reshape(-1, 1) / steadystate.sum()
    while diff > 1e-10:
        old_ss = steadystate.copy()
        steadystate = transition @ steadystate
        diff = np.abs(np.max(steadystate - old_ss))
        count += 1
        if count > 1e5:
            break

    diag = sparse.spdiags(
        steadystate.reshape((1, -1)), [0], node_num, node_num
    )
    if return_transition:
        return transition, diag, np.array(steadystate).flatten()
    else:
        return transition @ diag, np.array(steadystate).flatten()


def edgelist2csr_sparse(edgelist, node_num):
    """Edges as [(i, j, weight), …]"""
    graph = sparse.coo_matrix(
        (
            # data
            [e[2] for e in edgelist],
            # i and j
            ([e[1] for e in edgelist], [e[0] for e in edgelist]),
        ),
        shape=(node_num, node_num),
    )
    return sparse.csr_matrix(graph)


def partition2coo_sparse(part):
    """from dict {(i, j, k, …): weight, …}"""
    n_n = len(part)
    n_p = part.np
    try:
        return sparse.coo_matrix(
            (np.ones(n_n), (list(part.keys()), list(part.values()))),
            shape=(n_n, n_p),
            dtype=float,
        )
    except ValueError:
        print(n_p, n_n)
        raise


def kron(A, B):
    dok = {}
    for n in range(A.shape[0]):
        for pA in A.paths_through_node(n, position=-1):
            for pB in B.paths_through_node(n, position=0):
                dok[tuple(list(pA) + list(pB))] = A[pA] * B[pB]

    return SparseMat(dok)


def zeros(node_num):
    return SparseMat({}, node_num=0, normalize=False)


def zeros_like(sparsemat):
    return SparseMat(
        {}, node_num=sparsemat.nn, normalize=1.0, plength=sparsemat._dim
    )
