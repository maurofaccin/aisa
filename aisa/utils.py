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
"""Utility functions and classes."""

import numpy as np
from scipy import sparse

__all__ = ['Prob', "SparseMat", "entropy", "range_dependent_graph"]
np.seterr(all="raise")


class Prob():
    """A class to store the probability p and the plogp value."""

    __slots__ = ["__p", "__plogp"]

    def __init__(self, value):
        """Given a float or a Prob, store p and plogp.

        Parameters
        ----------
        value: float
            The probability value
        """
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
        """Return plog(p).

        Returns
        -------
        plogp : float
            return the values of plog(p)
        """
        return self.__plogp

    @property
    def p(self):
        """Return the value of p.

        Returns
        -------
        p : float
            the value of p
        """
        return self.__p

    def copy(self):
        """Return a copy of itself.

        Returns
        -------
        probability : Prob
            a copy of self
        """
        return Prob(self)

    def __float__(self):
        """Return probability.

        Returns
        -------
        p : float
            returns the probability as float.
        """
        return self.__p

    def __iadd__(self, other):
        """Add."""
        self.__p += float(other)
        self.__update_plogp()
        return self

    def __add__(self, other):
        """Add."""
        new = self.copy()
        new += other
        return new

    def __isub__(self, other):
        """Subtract."""
        self.__p -= float(other)
        self.__update_plogp()
        return self

    def __sub__(self, other):
        """Subtract."""
        new = self.copy()
        new -= other
        return new

    def __imul__(self, other):
        """Multiply."""
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
        """Multiply."""
        new = self.copy()
        new *= other
        return new

    def __itruediv__(self, other):
        """Divide."""
        self.__p /= float(other)
        self.__update_plogp()
        return self

    def __truediv__(self, other):
        """Divide."""
        new = self.copy()
        new /= other
        return new

    def __repr__(self):
        """Representation."""
        return "{:g} [{:g}]".format(self.__p, self.__plogp)

    def __eq__(self, other):
        """Test equivalence."""
        # TODO: add approx?
        return self.__p == float(other)

    # set inverse operators
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__


class SparseMat():
    """A sparse square matrix with column and row slicing capabilities.

    The aim of this matrix is to represent a probability distribution
    and facilitate entropy calculation.
    It may be of more that two dimensions.
    """

    __slots__ = ["_dok", "_nn", "_dim", "_norm", "__p_thr"]

    def __init__(self, mat, node_num=None, normalize=False, plength=None):
        """Initiate the matrix.

        Parameters
        ----------
        mat : scipy sparse matrix or list or dict
            The values of the matrix which can be:
                - a scipy sparse matrix
                - a list of ((i, j, ...), w)
                - a dict (i, j, ...): w
        node_num : int
            number of nodes
        normalize : bool
            whether to normalize entries or not
        plength : int
            lenght of each path, to use only if len(mat) == 0

        Raises
        ------
        ValueError :
            if normalize is not a float, Prob or bool
        """
        self._dok, self._nn, self._dim = self.__any_to_dok__(mat, node_num, plength=plength)

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

    @classmethod
    def __any_to_dok__(self, mat, node_num, plength=None):
        """Convert to dok."""
        if isinstance(mat, sparse.spmatrix):
            mat = sparse.coo_matrix(mat)
            dok = {
                (i, j): Prob(d) for i, j, d in zip(mat.col, mat.row, mat.data)
            }

            nn = node_num if node_num is not None else mat.shape[0]
            dim = 2
        elif isinstance(mat, dict):
            dok = {tuple(k): Prob(v) for k, v in mat.items()}
            if node_num is None:
                nn = np.max([dd for d in self._dok for dd in d]) + 1
            else:
                nn = node_num
            # get the first key of the dict
            if plength is None:
                val = next(iter(dok.keys()))
                dim = len(val)
            else:
                dim = plength
        else:
            dok = {tuple(i): Prob(d) for i, d in mat}
            if node_num is None:
                nn = np.max([dd for d in self._dok for dd in d]) + 1
            else:
                nn = node_num
            dim = len(mat[0][0])

        return dok, nn, dim

    def entropy(self):
        """Return the entropy of self.

        Returns
        -------
        entropy : float
            The entropy (assuming this matrix is a probability distribution)
        """
        if self._nn == 0:
            return 0.0
        sum_plogp = np.sum([p.plogp for p in self._dok.values()])
        return (self._norm.plogp - sum_plogp) / self._norm.p

    @property
    def shape(self):
        """Return the shapeof the matrix.

        Returns
        -------
        shape : tuple
            a tuple of ints representing the shape of the matrix
        """
        return tuple([self._nn] * self._dim)

    @property
    def nn(self):
        """Return the number of nodes.

        Returns
        -------
        nn : int
            the number of nodes
        """
        return self._nn

    def size(self):
        """Return the number of non-zero elements.

        Returns
        -------
        size : int
            the number of non-zero elements.
        """
        return len(self._dok)

    def project(self, part, move_node=None):
        """Project the matrix to a partition subspace.

        Parameters
        ----------
        part : dict
            a partition of the graph.
        move_node : tuple or None, default: None
            a tuple with (node, new_partition).
            If not None, this node is moved to the new partition before projecting.

        Returns
        -------
        projected_matrix : SparseMat
            the matrix projected to the partition space.
        """
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
        """Return a copy of itself.

        Returns
        -------
        sparse_mat : SparseMat
            a copy of itself.
        """
        return SparseMat(
            {path[:]: w.copy() for path, w in self._dok.items()},
            node_num=self._nn,
            normalize=self._norm,
            plength=self._dim,
        )

    def get_egonet(self, inode, axis=None):
        r"""Extract ego-network of a node.

        Parameters
        ----------
        inode : int
            node
        axis : int or None, default=None
            If not `None`, restricts the returned egonet to paths that pass through
            the given node at the `axis` step. Othewise all paths going through
            that node are returned.

        Returns
        -------
        egonet : SparseMat
            the egonet of the node.
        """
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
        """Return a submatrix of self defined by nodes in `inodes`.

        Parameters
        ----------
        inodes : list of ints
            nodes to extract from the matrix
        Returns
        -------
        submatrix : SparseMat
            the submatrix of all paths going through any of the nodes in `inodes`
        """
        return SparseMat(
            {
                p: self._dok[p]
                for p in set().union(*[self.__p_thr[i] for i in inodes])
            },
            node_num=self._nn,
            normalize=self._norm,
        )

    def slice(self, axis=0, n=0):
        """Return a slice along an axis.

        Parameters
        ----------
        axis : {0, 1}
            axis to slice (Default value = 0)
        n :
            (Default value = 0)

        Returns
        -------
        output : array
            description
        """
        def path(x, y, d):
            """Return the weight of constant path except for on an axis.

            Returns
            -------
            value :
                the entry of self corresponding to the path [y, y, y, y, x, y, y, y]
                where `x` is on axis `d`.
            """
            p = np.full(self._dim, y)
            p[d] = x
            return self._dok.get(tuple(p), 0.0)

        return np.array([path(n, nn, axis) for nn in range(self._nn)])

    def get_random_entry(self, return_all_probs=False):
        """Get a random entry weighted by the weights.

        Parameters
        ----------
        return_all_probs : (optional) bool, default=False
            if True return the probability of choosing any path

        Returns
        -------
        path : tuple
            the path
        link_prob : float
            the probability of choosing this path
        probs : (optional) array
            if `return_all_probs` is True, the probabilities to choose any path.
        """
        probs = np.array([float(n) for n in self._dok.values()])
        probs /= probs.sum()

        # choose one neighbour based on probs
        link_id = np.random.choice(len(self._dok), p=probs)
        link_prob = probs[link_id]
        link = list(self._dok.keys())[link_id]
        if return_all_probs:
            return link, link_prob, probs
        return link, link_prob

    def paths_through_node(self, node, axis=0):
        """Retuns all paths going through a node at a given step.

        Parameters
        ----------
        node : int
            the node
        axis : int, default=0
            The step at which the path should go through the node.

        Returns
        -------
        paths : list of tuples
            the list of paths going through the node at the given step.
        """
        return [p for p in self.__p_thr[node] if p[axis] == node]

    def paths(self, axis=None, node=None):
        """Iterate over paths.

        Parameters
        ----------
        axis : (optional) int or None
            if not None, return the paths that go through the node at this step.
        node : (optional) int or None
            if not None, return the paths that go through the node at this step.

        Yields
        ------
        paths, value : tuple, Prob
            paths and values. If both `axis` and `node` are not None, only paths
            going through that node at that step.
        """
        if axis is None or node is None:
            yield from self.__iter__()

        else:
            for p, v in self._dok.items():
                if p[axis] == node:
                    yield p, v / self._norm

    def set_path(self, path, weight):
        """Overwrite path weight.

        Parameters
        ----------
        path : tuple of ints
            the new path
        weight : float or Prob
            the new weight
        """
        self._dok[path] = Prob(weight) * self._norm
        for i in path:
            self.__p_thr[i].add(path)

    def get_from_sparse(self, other, normalize=False):
        """Get weights from self on entries from another SparseMat.

        Parameters
        ----------
        other : SparseMat
            the matrix defining the paths of interest
        normalize : (optional) bool, default=False
            if True, renormalize before returning.

        Returns
        -------
        matrix : SparseMat
            the new matrix with entries from self where other is non-zero.
        """
        return SparseMat(
            {p: self._dok[p] for p, _ in other if p in self._dok},
            node_num=self._nn,
            normalize=normalize,
        )

    def get_from_paths(self, paths, normalize=False):
        """Get weights from self on entries from a list of paths.

        Parameters
        ----------
        paths : list of tuples of ints
            a list of paths.
        normalize : (optional) bool, default=False
            if True, renormalize before returning.

        Returns
        -------
        matrix : SparseMat
            the new matrix with entries only from the list of paths.
        """
        return SparseMat(
            {p: self._dok[p] for p in paths if p in self._dok},
            node_num=self._nn,
            normalize=normalize,
        )

    def add_colrow(self):
        """Add a new empty node at the end."""
        self._nn += 1
        self.__p_thr.append(set())
        return self._nn - 1

    def merge_colrow(self, index_from, index_to):
        """Merge two indexes in each dimension.

        Parameters
        ----------
        index_from : int
            index to merge (will be removed)
        index_to : int
            index to merge.

        Returns
        -------
        new_matrix : SparseMat
            new matrix with `index_to` and `index_from` merged.
        """
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
        """Kronecker product."""
        dok = {}
        for n in range(self._nn):
            for pA in self.paths_through_node(n, axis=-1):
                for pB in other.paths_through_node(n, axis=0):
                    dok[pA[:-1] + pB] = self._dok[pA] * other._dok[pB]

        return SparseMat(
            dok,
            node_num=self._nn,
            normalize=self._norm * other._norm
            # normalize=True
        )

    def sum(self, axis=None):
        """Retuns the sum of all entries.

        Parameters
        ----------
        axis : (optional) int or None, default=None
            if not `None`, project to the given axis (or step).

        Returns
        -------
        sum : float
        """
        # return the sum of all entries
        if axis is not None:
            probs = np.zeros(self._nn)
            for p, v in self._dok.items():
                probs[p[axis]] += float(v)
            return probs / float(self._norm)
        if self._nn == 0:
            return 0.0
        if len(self._dok) == 0:
            return 0.0
        return np.sum([float(p) for p in self._dok.values()]) / float(
            self._norm
        )

    def __update_all_paths(self):
        """For each node, all paths that go through it."""
        self.__p_thr = [set() for _ in range(self._nn)]
        for path in self._dok.keys():
            for i in path:
                try:
                    self.__p_thr[i].add(path)
                except IndexError:
                    print(path, self._nn)
                    raise

    def __or__(self, other):
        """Return a SparseMat with entries from both self and other.

        Local entries will be overwritten by other's.
        """
        new = self.copy()
        for p, v in other:
            new.set_path(p, v)
        return new

    def __iter__(self):
        """Iterate over paths and values."""
        for k, v in self._dok.items():
            yield k, v / self._norm

    def __getitem__(self, path):
        """Get an item."""
        try:
            return self._dok[path] / self._norm
        except KeyError:
            return 0.0

    def __iadd__(self, other):
        """Add."""
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
        """Add."""
        new = self.copy()
        new += other
        return new

    def __isub__(self, other):
        """Subtract.

        Can provide negative values.
        """
        ratio = self._norm / other._norm
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
        """Subtract."""
        new = self.copy()
        new -= other
        return new

    def __imul__(self, other):
        """Multiply."""
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
        """Multiply."""
        new = self.copy()
        new *= other
        return new

    def __eq__(self, other):
        """Test equivalence."""
        for p, v in self._dok.items():
            if not np.isclose(
                float(v / self._norm),
                float(other._dok[p] / other._norm),
                atol=1e-10,
            ):
                return False
        return True

    def to_csr(self):
        """Return a scipy sparse matrix.

        It will be projected to the first two axis.
        """
        smat = sparse.coo_matrix(
            (
                [float(p) for p in self._dok.values()],
                (
                    [indx[0] for indx in self._dok],
                    [indx[1] for indx in self._dok]
                )
            ),
            shape=self.shape,
            dtype=float
        )
        return smat.tocsr()


class Partition():
    """a bidirectional dictionary to store partitions. (for internal use)."""

    def __init__(self, partition: dict):
        """Build a dictionary node->class."""
        if partition is None:
            self.n2i = None
            self.i2n = None
            self.parts = None
            self.partition = {}
        else:
            self.from_dictionary(partition)

    def node_names(self):
        """Return the map of node names to node indexes."""
        yield from self.n2i

    def to_dictionary(self):
        """Return a dictionary copy of self."""
        return {self.i2n[inode]: part for inode, part in self.items()}

    def from_dictionary(self, partition):
        """Set partition from a node->class dictionary.

        Parameters
        ----------
        partition : dict
        """
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
        """Return a coo sparse matrix of the projector."""
        return partition2coo_sparse(self)

    @property
    def np(self):
        """Retun the number of partitions."""
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
        self.parts[old_part].remove(node)

    def __len__(self):
        return len(self.partition)

    def __delitem__(self, node):
        """Remove nodes."""
        raise AttributeError('Nodes cannot be removed')

    def merge(self, part_from, part_to):
        """Merge two classes."""
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
        """Iterate over node, partition tuples."""
        yield from self.partition.items()

    def __getitem__(self, node):
        return self.partition[node]

    def copy(self):
        """Return a copy of self."""
        return Partition(self.to_dictionary())

    def keys(self):
        """Iterate over node names."""
        yield from self.partition.keys()

    def values(self):
        """Iterate over classes."""
        yield from self.partition.values()


def entropy(array):
    """Compute the entropy of the array (or float).

    Parameters
    ----------
    array : float, SparseMat, np.array
        An array or a float.

    Returns
    -------
    entropy : float
        The entropy of the array
    """
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
    r"""Compute p_ij and p_i at the steady state.

    Parameters
    ----------
    edges : list of tuples
        list of edges/paths
    node_num : int
        number of nodes
    symmetric : (optional) bool (Default value = False)
        if True use the edges also backwards
    return_transition: (optional) bool (Default value = False)
        if True, returns also the transition matrix.
    compute_steady : (optional) bool (Default value = False)
    T : (optional) int or None (Default value = None)
        temporal parameter.

    Returns
    -------
    p_ij, p_i : sparse, array
        Probabilities
        If `return_transition` is True, returns a tuple with transition matrix,
        a diagonal matrix with the inverse of steadystate entries, steadystate.
    """
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

    return transition @ diag, np.array(steadystate).flatten()


def edgelist2csr_sparse(edgelist, node_num):
    """Return a sparse matrix given a list of edges.

    Parameters
    ----------
    edgelist : list of tuples of ints
        the list of edges
    node_num : int
        number of nodes.

    Returns
    -------
    adjacency_matrix : csr_sparse
        the adjacency matrix as csr_sparse matrix
    """
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
    """Return a coo_sparse matrix given a dict {(i, j, k, …): weight, …}."""
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
    """Kronecker."""
    dok = {}
    for n in range(A.shape[0]):
        for pA in A.paths_through_node(n, position=-1):
            for pB in B.paths_through_node(n, position=0):
                dok[tuple(list(pA) + list(pB))] = A[pA] * B[pB]

    return SparseMat(dok)


def range_dependent_graph(nnodes, alphas, gammas, symmetric=False):
    """Range depentent graph.

    Parameters
    ----------
    nnodes : list of ints
        nodes of each class (len(nodes) = k)
    alphas : matrix-like
        matrix of parameter alpha (k x k)
    gammas : matrix-like
        matrix of parameter gamma (k x k)
    symmetric : bool, default=False
        force out graph to be symmetric (Default value = False)

    Returns
    -------
    graph : nx.Graph or nx.DiGraph
        the graph
    """
    alphas = np.asarray(alphas)
    gammas = np.asarray(gammas)

    blocks = [[None for _ in nnodes] for _ in nnodes]

    for inn, nn in enumerate(nnodes):
        blocks[inn][inn] = range_dependent_block(
            nn,
            alphas[inn, inn],
            gammas[inn, inn],
            symmetric=symmetric
        )

    for i, j in zip(*np.triu_indices(len(nnodes), k=1)):
        block = range_dependent_block(
            (nnodes[i], nnodes[j]),
            alphas[i, j],
            gammas[i, j],
            symmetric=False
        )

        blocks[i][j] = block

        if symmetric:
            block = block.T
        else:
            block = range_dependent_block(
                (nnodes[j], nnodes[i]),
                alphas[i, j],
                gammas[i, j],
                symmetric=False
            )
        blocks[j][i] = block

    return sparse.bmat(blocks, format='csr')


def range_dependent_block(nnodes, alpha, gamma, symmetric=False):
    r"""Range depentent block.

    The probability is given by:

    \( p_{ij} = \alpha \gamma^{d_{ij}} \)

    where

    \( d_{ij} = \frac{d_\theta (i, j )}{2\pi} \sqrt{N_{c_i} N_{c_j}} \)

    where \( d_\theta (j,i) \) is the shorter angular path.

    Parameters
    ----------
    nnodes : int or tuple of two ints
        nodes of each class (len(nodes) = k)
    alpha : float
        parameter alpha
    gamma : float
        parameter gamma
    symmetric: bool, default=False
        force out graph to be symmetric (Default value = False)

    Returns
    -------
    graph : sparse matrix
        matrix
    """
    # random number genertor
    rng = np.random.default_rng()

    # get the adjacency matrix
    if isinstance(nnodes, (int, np.integer)):
        nnodes = (nnodes, nnodes)
    assert len(nnodes) == 2, "For now we expect two dimensions!"
    adj = sparse.lil_matrix(nnodes, dtype=np.int8)

    ind0, ind1 = np.indices(nnodes, dtype=np.float32)

    ind0 = ind0 / nnodes[0]
    ind1 = ind1 / nnodes[1]

    dtheta = np.abs(ind0 - ind1)
    dtheta = np.minimum(dtheta, np.abs(dtheta - 1))
    dij = dtheta * np.sqrt(nnodes[0] * nnodes[1])

    pij = alpha * gamma ** dij
    np.fill_diagonal(pij, 0.0)

    adj = (rng.random(size=nnodes) <= pij).astype(int)
    adj = sparse.csr_matrix(adj)

    if nnodes[0] != nnodes[1] or not symmetric:
        return adj

    return sparse.tril(adj) + sparse.tril(adj, -1).T
