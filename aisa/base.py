#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# File      : base.py
# Creation  : 01 Sept 2020
#
# Copyright (c) 2020 Mauro Faccin <mauro.fccn@gmail.com>
#               https://maurofaccin.github.io
#
# Description : Module to aggregate states of a dynamical system
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

from collections import Counter
import logging
import sys
import itertools

import numpy as np
import networkx as nx
from scipy import sparse
from . import utils

try:
    import tqdm
except ModuleNotFoundError:
    pass

__all__ = ["PGraph", "best_partition", "merge_pgraph", "entrogram", "optimize"]

FORMAT = "%(asctime)-15s || %(message)s"
logging.basicConfig(format=FORMAT)
log = logging.getLogger("EntroLog")
log.setLevel(logging.WARNING)
SYMBOLS = "0123456789ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghjklmnopqrstuvwxyz"


class PGraph():
    """A graph with a partition.
    """

    def __init__(self, graph, init_part=None, compute_steady=True, T=None):
        """Initialize from `networkx` graphs.

        Parameters
        ---

        graph: nx.[Di]Graph()
            The graph

        init_part: dict
            initial partition to start the optimization. (Default: N singletons)

        compute_steady: bool
            If steady state need to be computed. (Default: True)
            If False, the steady state will be the marginal.

        T: int
            time scale parameter value. (Default: 1)
        """

        if isinstance(graph, nx.DiGraph):
            self._isdirected = True
        elif isinstance(graph, nx.Graph):
            self._isdirected = False
        else:
            raise ValueError("Need nx.[Di]Graph, not " + str(type(graph)))

        _graph = nx.DiGraph(graph)
        nnodes = _graph.number_of_nodes()

        if init_part is None:
            init_part = {node: node for node in range(_graph.number_of_nodes())}
        self._part = utils.Partition(init_part)

        # compute the probabilities p(i, j) and p(i)
        edges = [
            (self._part.n2i[i], self._part.n2i[j], w)
            for i, j, w in _graph.edges.data("weight", default=1.0)
        ]
        p_ij, p_i = utils.get_probabilities(
            edges,
            nnodes,
            symmetric=not self._isdirected,
            return_transition=False,
            compute_steady=compute_steady,
            T=T
        )
        self._pij = utils.SparseMat(p_ij, normalize=True)
        self._pi = p_i / p_i.sum()

        # save partition and projected p(i,j) and p(i)
        self.set_partition()

        assert np.isclose(self._pij.sum(), 1.0)
        assert np.isclose(self._ppij.sum(), 1.0)

        self._reset()

    def set_partition(self, partition=None):
        """Set/Change the graph partition.

        Parameters
        ---

        partition: dict
            The partition to be set. A dict with nodes as keys and classes as values

        """
        # set up partition
        if partition is not None:
            self._part = utils.Partition(partition)
        _part = sparse.csr_matrix(self._part.to_coo())
        self._nn, self._np = _part.shape

        self._ppij = self._pij.project(self._part)
        p_pi = _part.transpose() @ self._pi
        self._ppi = p_pi / p_pi.sum()

    @property
    def np(self):
        """Number of partitions"""
        return self._np

    @property
    def nn(self):
        """Number of nodes"""
        return self._nn

    def _move_probability(self, inode, part_probs=None, part=None):
        """Compute the probability to move to a partition.

        :inode: node index to move
        :part_probs: optional prob distribution to compute from
        :part: optional partition to move the node to
        """

        if part_probs is None:
            part_probs = self._ppij.copy()
        n_ego_out = self._pij.get_egonet(inode, axis=0).project(self._part)
        n_ego_in = self._pij.get_egonet(inode, axis=1).project(self._part)

        # compute probability to choose a partition to mode to
        if part is None:
            probs = np.zeros(part_probs.nn)
            rng = range(part_probs.nn)
        else:
            probs = np.array([0.0])
            rng = [part]

        for prt in rng:

            probs[prt] += np.sum(
                [
                    float(v) * float(part_probs[(prt, p[1])])
                    for p, v in n_ego_out
                ]
            )

            probs[prt] += np.sum(
                [
                    float(v) * float(part_probs[(p[0], prt)])
                    for p, v in n_ego_in
                ]
            )

        probs += 1.0 / (part_probs.nn + 1)
        return probs / probs.sum()

    def _get_random_move(self, inode=None, kmax=None, kmin=None, **kwargs):
        """Select one node and a partition to move to.
        Returns the probability of the move and the delta energy.
        """
        rdm = np.random.default_rng()

        # get a random node
        if inode is None:
            inode = rdm.integers(self._nn)
        # save its starting partition
        old_part = self._part[inode]

        if kmax is not None and self._np == kmax:
            # do not go for a new partition
            delta = 0.0
        else:
            delta = 1.0 / (self._nn + 1)

        if kmin is not None:
            if len(self._part.parts[old_part]) == 1 and self._np == kmin:
                # do not move this node, it's the last one
                return inode, None, None, None

        n_ego_full = self._pij.get_egonet(inode)
        if n_ego_full is None:
            return None, None, None, None
        n_ego = n_ego_full.project(self._part)

        if rdm.random() < delta:
            # inode is going to start a new partition
            n_ego.add_colrow()
            p_sub = self._ppij.get_egonet(old_part)
            p_sub.add_colrow()
            new_part = self._np

            prob_move = delta
            act = "split"
        else:
            # move inode to anothere partition
            probs_go = self._move_probability(inode)
            new_part = rdm.choice(np.arange(self._np), p=probs_go)
            p_sub = self._ppij.get_submat([old_part, new_part])

            prob_move = probs_go[new_part]
            if len(self._part.parts[old_part]) == 1:
                act = "merge"
            else:
                act = "move"

        if (inode, new_part) in self._tryed_moves:
            return (
                inode,
                new_part,
                self._tryed_moves[(inode, new_part)][0],
                self._tryed_moves[(inode, new_part)][1],
            )

        n_ego_post = n_ego_full.project(
            self._part,
            move_node=(inode, new_part)
        )

        H2_pre = utils.entropy(p_sub)
        # I make this in two steps in order to get alway positive values
        p_sub += n_ego_post
        p_sub -= n_ego
        H2_post = utils.entropy(p_sub)

        probs_back = self._move_probability(inode, part_probs=p_sub)
        prob_ratio = probs_back[old_part] / prob_move

        if new_part == self._np:
            h1_pre = utils.entropy(self._ppi[old_part])
            h1_post = utils.entropy(
                [self._ppi[old_part] - self._pi[inode], self._pi[inode]]
            )
        else:
            h1_pre = utils.entropy([self._ppi[old_part], self._ppi[new_part]])
            h1_post = utils.entropy(
                [
                    self._ppi[old_part] - self._pi[inode],
                    self._ppi[new_part] + self._pi[inode],
                ]
            )

        delta_obj = self.__delta__(
            h1_pre, H2_pre, h1_post, H2_post, action=act, **kwargs
        )

        self._tryed_moves[(inode, new_part)] = (prob_ratio, delta_obj)
        return inode, new_part, prob_ratio, delta_obj

    def _move_node(self, inode, partition):
        if self._part[inode] == partition:
            return None

        old_part = self._part[inode]
        if len(self._part.parts[old_part]) == 1:
            self.merge_partitions(partition, old_part)
            return None

        pnode = self._pi[inode]
        self._ppi[old_part] -= pnode
        self._ppi[partition] += pnode

        ego_node = self._pij.get_egonet(inode)
        proj_ego_org = ego_node.project(self._part)
        proj_ego_dst = ego_node.project(
            self._part, move_node=(inode, partition)
        )
        self._ppij += proj_ego_dst
        self._ppij -= proj_ego_org

        self._part[inode] = partition
        self._reset()

    def _get_best_merge(self, checks=200, **kwargs):
        best = {"parts": (None, None), "delta": -np.inf}
        if self._np <= 1:
            return None

        for p1, p2 in neigneig_full(self):
            d = self.__try_merge__(p1, p2, **kwargs)
            if d > best["delta"]:
                best = {"parts": (p1, p2), "delta": d}
            if d > 0:
                break
            checks -= 1
            if checks < 1:
                break

        return best["parts"]

    def __try_merge__(self, p1, p2, **kwargs):
        p12 = self._ppij.get_submat([p1, p2])
        H2pre = utils.entropy(p12)

        p12 = p12.merge_colrow(p1, p2)
        H2post = utils.entropy(p12)

        h1pre = utils.entropy(self._ppi[p1]) + utils.entropy(self._ppi[p2])
        h1post = utils.entropy(self._ppi[p1] + self._ppi[p2])
        return self.__delta__(
            h1pre, H2pre, h1post, H2post, action="merge", **kwargs
        )

    def _split(self, inode):
        old_part = self._part[inode]
        log.debug("Splitting node {}".format(inode))
        if len(self._part.parts[old_part]) == 1:
            return

        # self._ppi
        self._ppi = np.append(self._ppi, [0])
        self._ppi[old_part] -= self._pi[inode]
        self._ppi[-1] = self._pi[inode]

        # self.__ppij
        new_part = self._ppij.add_colrow()

        ego_node = self._pij.get_egonet(inode)
        en_pre = ego_node.project(self._part)
        # self._part
        self._part[inode] = new_part
        en_post = ego_node.project(self._part)
        # make this in two steps in order to have always non negative values
        self._ppij += en_post
        self._ppij -= en_pre

        # final updates
        self._np += 1
        self._reset()

    def merge_partitions(self, part1, part2):
        """Merge partitions into one.

        Merge partition `part1` and `part2` and update pgraph accordingly.

        Parameters
        ---

        part1: int
            integer index of the first partition to be merged

        part2: int
            integer index of the second partition to be merged
        """
        log.debug("Merging partitions {} and {}.".format(part1, part2))
        part_to, part_from = sorted([part1, part2])

        # self._ppi
        self._ppi[part_to] += self._ppi[part_from]
        self._ppi = np.array(
            [self._ppi[i] for i in range(self._np) if i != part_from]
        )

        # self.__ppij
        self._ppij = self._ppij.merge_colrow(part_from, part_to)

        # self._part
        self._part.merge(part_from, part_to)

        self._np -= 1
        self._reset()

    def _reset(self):
        self._ppij = self._pij.project(self._part)
        self._tryed_moves = {}

    def nodes(self):
        """Iterator over nodes names.

        Yields
        ---

        nodes
        """
        yield from self._part.node_names()

    def __repr__(self):
        return "Graph with {} nodes {} edges and {} partitions".format(
            self._nn, len(self._pij._dok), self._np
        )

    def print_partition(self):
        """
        Try to print the partition to screen using ASCII chars.
        """
        try:
            strng = "".join([SYMBOLS[self._part[i]] for i in range(self._nn)])
        except IndexError:
            return "Too many symbols!"
        if len(strng) > 80:
            return strng[:78] + "…"
        else:
            return strng

    def partition(self):
        """Return the partition.

        Returns
        ---

        partition: dict
            a dictionary with nodes as keys and classes as values.
        """

        return self._part.to_dictionary()

    def __delta__(
        self, h1old, h2old, h1new, h2new, beta=0.0, action="move"
    ):
        return (2 - beta) * (h1new - h1old) - h2new + h2old

    def autoinformation(self, beta):
        """Return the autoinformation value for the current partion.

        Parameters
        ---

        beta: float
            model selection parameter. (Default: 0)

        Returns
        ---

        autoinformation: float
            the autoinformation relative to the graph, partition and \( \\beta \)
        """
        h_1 = utils.entropy(self._ppi)
        h_2 = utils.entropy(self._ppij)
        return (2 - beta) * h_1 - h_2


def neigneig_full(pgraph, kind='projected'):
    """Return the set of neighbours of neighbours of node."""

    if kind == "projected":
        s_mat = pgraph._ppij
    elif kind == "full":
        s_mat = pgraph._pij
    else:
        raise ValueError("kind can be only 'projected' or 'full'")

    inpairs = [set() for _ in range(s_mat.nn)]
    outpairs = [set() for _ in range(s_mat.nn)]

    for (p1, p2), v in s_mat.paths():
        inpairs[p1].add(p2)
        outpairs[p2].add(p1)

    pairs = set()
    for s in inpairs + outpairs:
        pairs |= set(itertools.combinations(sorted(s), 2))

    return pairs


def entrogram(graph, partition, depth=3):
    """Compute the entrogram for the graph and the given partition.
    A random walk is assumed as Markovian process on the original network.

    Parameters
    ---

    graph: nx.Graph or nx.DiGraph

    partition: dict
        A dictionary with nodes as keys and values as classes.

    depth: int
        The number of bars the final entrogram should have. (Default: 3)

    Returns
    ---

    entrogram: tuple
        - \( H_{KS} \)
        - list of entrogram entries

    """
    # node to index map
    part = utils.Partition(partition)
    n_n = len(part)
    n_p = part.np

    symmetric = True if isinstance(graph, nx.Graph) else False
    edges = [
        (part.n2i[nodei], part.n2i[nodej], weight)
        for nodei, nodej, weight in graph.edges.data("weight", default=1.0)
    ]
    if symmetric:
        edges += [(j, i, w) for i, j, w in edges]

    transition, diag, pi = utils.get_probabilities(
        edges, n_n, symmetric=symmetric, return_transition=True
    )

    pij = transition @ diag
    pij = utils.SparseMat(pij, normalize=True)
    # transition[i, j] = p(j| i)
    transition = utils.SparseMat(transition)

    p_pij = pij.project(part)
    p_pi = np.zeros(n_p)
    for (i, j), w in p_pij:
        p_pi[i] += w
    Hs = [utils.entropy(p_pi), utils.entropy(p_pij)]

    for step in range(1, depth + 1):
        # pij = utils.kron(pij, transition)
        pij = pij.kron(transition)
        p_pij = pij.project(part)
        Hs.append(utils.entropy(p_pij))

    entrogram = np.array(Hs)
    entrogram = entrogram[1:] - entrogram[:-1]
    Hks = Hs[-1] - Hs[-2]
    return Hks, entrogram[:-1] - Hks


def best_partition(
            graph,
            T=None,
            beta=0.0,
            init_part=None,
            kmin=None,
            kmax=None,
            invtemp=1e6,
            compute_steady=True,
            tsteps=10000,
            ):
    """Find the best partition for `graph`.

    Parameters
    ----------

    graph: nx.Graph or nx.DiGraph
        The graph to be aggregated (a random walk is considered as dynamical system)

    init_part: dict
        initial partition to start the optimization. (Default: N singletons)

    T: int
        time scale parameter value. (Default: 1)

    beta: float
        model selection parameter. (Default: 0)

    kmin: int
        minimum number of partition to be accepted

    kmax: int
        maximum number of partition to be accepted

    invtemp: float
        the inverse of the pseudo-temperature for the simulated annealing process

    compute_steady: bool
        If steady state need to be computed. (Default: True)
        If False, the steady state will be the marginal.

    tsteps: int
        Maximum number of steps in the optimization process. (Default: 10k)


    Returns
    ------

    partition: dict
        Dictionary with nodes as keys and partitions as values.

    """

    if kmax is None:
        kmax = graph.number_of_nodes()

    if kmin is None:
        kmin = 1

    if init_part is None:
        # start from N partitions
        initp = {n: i for i, n in enumerate(graph.nodes())}
    elif isinstance(init_part, dict):
        initp = init_part
    else:
        raise ValueError(
            "init_part should be a dict not {}".format(type(init_part))
        )

    pgraph = PGraph(graph, compute_steady=compute_steady, init_part=initp, T=T)

    log.info(
        "Optimization with {} parts, beta {}, 1/T {}".format(
            pgraph._np, beta, invtemp
        )
    )

    # start with hierarchical merging
    merge_pgraph(pgraph, kmin=kmin, kmax=kmax, beta=beta)
    # optimize
    best = optimize(pgraph, invtemp, tsteps, kmin, kmax, beta=beta)

    results = dict(best)
    pgraph.set_partition(results)
    autoinformation = pgraph.autoinformation(beta)

    log.info("final: num part {}".format(pgraph.np))
    log.info("{} -- {} ".format(pgraph._np, pgraph.print_partition()))
    log.info("   -- {}".format(autoinformation))

    return results


def optimize(pgraph, kmin, kmax, invtemp, tsteps, beta=0.0):
    """
    Optimize the partition enbedded into `pgraph`.

    Parameters
    ---

    pgraph: PGraph
        A graph plus partition.

    kmin: int
        minimum number of partition to be accepted

    kmax: int
        maximum number of partition to be accepted

    invtemp: float
        the inverse of the pseudo-temperature for the simulated annealing process

    tsteps: int
        Maximum number of steps in the optimization process. (Default: 10k)
    """

    rdm = np.random.default_rng()

    bestp = pgraph.partition()
    cumul = 0.0
    moves = [
        0,  # delta > 0
        0,  # delta < 0 accepted
        0,  # best
        0,  # changes since last move
    ]

    if "tqdm" in sys.modules and log.level >= 20:
        tsrange = tqdm.trange(tsteps)
    else:
        tsrange = range(tsteps)

    for tstep in tsrange:
        r_node, r_part, p, delta = pgraph._get_random_move(kmin=kmin, kmax=kmax, beta=beta)

        if r_part is None:
            continue

        log.debug(
            "proposed move: n {:5}, p {:5}, p() {:5.3f}, d {}".format(
                r_node, r_part, p, delta
            )
        )

        log.debug("CUMUL {}".format(cumul))
        if delta is None:
            continue

        if delta >= 0.0:
            if r_part == pgraph.np:
                pgraph._split(r_node)
            else:
                # move or merge
                pgraph._move_node(r_node, r_part)
            cumul += delta
            moves[0] += 1
            moves[3] = 0
            log.debug("accepted move")
            if "tqdm" in sys.modules and log.level >= 20:
                tsrange.set_description("{} [{}]".format(moves[2], pgraph.np))
        else:
            rand = rdm.random()
            if rand == 0.0:
                continue

            sim_ann = 1.0 - tstep / tsteps
            if sim_ann == 0.0:
                continue
            threshold = invtemp * delta / sim_ann + np.log(p)
            if np.log(rand) < threshold:
                if r_part == pgraph.np:
                    pgraph._split(r_node)
                else:
                    pgraph._move_node(r_node, r_part)
                cumul += delta
                moves[1] += 1
                moves[3] = 0
                log.debug("accepted move {} < {}".format(rand, threshold))
                if "tqdm" in sys.modules and log.level >= 20:
                    tsrange.set_description(
                        "{} [{}]".format(moves[2], pgraph.np)
                    )
            else:
                log.debug("rejected move")
                moves[3] += 1

        if cumul > 0:
            log.debug("BEST move +++ {} +++".format(cumul))
            bestp = pgraph.partition()
            cumul = 0.0
            moves[2] += 1

        if moves[3] > 500:
            break

    log.info("good {}, not so good {}, best {}".format(*moves))
    return bestp


def merge_pgraph(pgraph, beta=0.0, kmin=1, kmax=np.inf):
    """Hierarchical merging of classes.

    Parameters
    ---

    pgraph: PGraph
        A graph plus partition.

    beta: float
        Model selection parameter (Default: 0.0)

    kmin: int
        minimum number of partition to be accepted

    kmax: int
        maximum number of partition to be accepted


    `pgraph` will be updated to the best encounteded partition.
    """

    if kmin > pgraph.np:
        return None

    elif pgraph.np <= kmax:
        best_part = (pgraph.autoinformation(beta), pgraph.partition())

    else:
        best_part = (-np.inf, None)

    while pgraph.np > kmin:
        print(pgraph.np, ' ', end='\r')
        best = pgraph._get_best_merge(beta=beta)
        if best is None:
            break
        pgraph.merge_partitions(*best)
        val = pgraph.autoinformation(beta)
        if val > best_part[0] and pgraph.np <= kmax:
            best_part = (val, pgraph.partition())

    pgraph.set_partition(best_part[1])
