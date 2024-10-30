import random
from collections import deque
from itertools import permutations

import networkx as nx
import numpy
import torch as T
import numpy as np
from .utils.buffer import ReplayBuffer
from warnings import simplefilter

from pgmpy.models import BayesianNetwork
from pgmpy.metrics import log_likelihood_score
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)

from sklearn.metrics import mutual_info_score

simplefilter(action="ignore", category=FutureWarning)

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

if T.backends.mps.is_available():
    device = T.device("mps")


class K2Agent:
    # This class should include the K2 Algorithm and the memorizing functions
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3, max_size=1000000, data=None, label=None, greedy=1, log=True):
        # gamma 0.8 0.9

        # Q(S,A) + a*(r*maxQ(S',A')-Q(S,A))
        self.alpha = alpha  # The Learning Rate
        self.gamma = gamma  # The Discount Factor
        self.e_epsilon = epsilon  # The Possibility for Exploration
        self.greedy = greedy  # The binary indicator for using greedy or random choice for exploration
        self.max_size = max_size  # The max_size for Q-table
        self.data = data  # For pgmpy part, need modification
        self.X = data.iloc[:, :-1]  # The dataset for the Bayesian Network Learning
        self.Y = label  # The label for the dataset for the Bayesian Network Learning
        self.variables = list(self.X.columns.values)  # The variables in the Baysian Network
        self.done_bonus = 0.3  # The bonus of finishing forming up the Bayesian Structural Learning
        self.log = log

        self.tabu_list = deque(maxlen=1000)

        # self.current_model = BayesianNetwork() # 就是Bayesian Network，但是要使用特定的node ordering进行观察
        # self.current_model.add_nodes_from(self.variables)

        self.node_list = self.ordering_nodes()

        # The Q-Table: Memorize the experience of Q(S,A)： S is the networkx object, the A is generated by choose_action
        # self.Q_table = numpy.ndarray(len(self.node_list),len(self.node_list))
        self.Q_table = {}

    def remember(self, state, action, reward, state_):
        # Calculate maxQ(s',a'),exploration = False for maximum
        # action_ = self.estimate_once(start_dag=state_)
        # Update the Q-value

        # Q(s,a) = Q(s,a) + a * (G_i - Q(s,a))
        self.Q_table[(state, action)] = self.Q_table.get((state, action), 0) + self.alpha * (reward -
                                                                                             self.Q_table.get(
                                                                                                 (state, action), 0))

        if self.log:
            print(self.Q_table)

    def ordering_nodes(self):
        mutual_info_dict = {}
        for node in self.variables:
            mi_score = mutual_info_score(self.X[node], self.Y)
            mutual_info_dict[node] = mi_score

        ordering = list(sorted(mutual_info_dict.items(), key=lambda item: item[1], reverse=True))
        keys_only = [item[0] for item in ordering]

        return keys_only

    def _available_actions(self, graph):  # the graph should be a Bayesian Network
        available_list = []
        for i, X in enumerate(self.node_list):
            for Y in self.node_list[i + 1:]:
                if not graph.has_edge(Y, X):
                    available_list.append((Y, X))
                # else:
                #     available_list.append(("-",(X,Y)))

        return available_list

    def _legacy_legal_operations(
            self,
            model,
            score,
            structure_score,
            max_indegree,
            black_list,
            white_list,
            fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip babilistic Graphical Mp a single edge. For details on scoring
        see Koller & Friedman, Proodels, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        # tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
                set(permutations(self.variables, 2))
                - set(model.edges())
                - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                        (operation not in self.tabu_list)
                        and ((X, Y) not in black_list)
                        and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        # - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        # for X, Y in model.edges():
        #     operation = ("-", (X, Y))
        #     if (operation not in self.tabu_list) and ((X, Y) not in fixed_edges):
        #         old_parents = model.get_parents(Y)
        #         new_parents = [var for var in old_parents if var != X]
        #         score_delta = score(Y, new_parents) - score(Y, old_parents)
        #         score_delta += structure_score("-")
        #         yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                    map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                        ((operation not in self.tabu_list) and ("flip", (Y, X)) not in self.tabu_list)
                        and ((X, Y) not in fixed_edges)
                        and ((Y, X) not in black_list)
                        and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = [var for var in old_Y_parents if var != X]
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                                score(X, new_X_parents)
                                + score(Y, new_Y_parents)
                                - score(X, old_X_parents)
                                - score(Y, old_Y_parents)
                        )
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)

    def _legal_operations(
            self,
            model,
            score,
            structure_score,
            max_indegree,
            black_list,
            white_list,
            fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip babilistic Graphical Mp a single edge. For details on scoring
        see Koller & Friedman, Proodels, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        # tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(self._available_actions(graph=model))
            # - set(model.edges()) # Only the non-exist edge are added following the ordering
            # - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.

            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                        (operation not in self.tabu_list)
                        and ((X, Y) not in black_list)
                        and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        # - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

    def estimate_once(
            self,
            scoring_method="bicscore",
            start_dag=None,
            fixed_edges=set(),
            max_indegree=None,
            black_list=None,
            white_list=None,
            epsilon=1e-4,
            custom_Q_table=None,
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore, aicscore. Also accepts a
            custom score, but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None

        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        Returns
        -------
        Estimated model: pgmpy.model.BayesianNetwork
            A `Bayesian Network` at a (local) score maximum.

        Examples
        --------
        """

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
        }

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        # if self.use_cache:
        if False:
            score_fn = ScoreCache.ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        # tabu_list = deque(maxlen=tabu_length)
        # tabu_list = self.tabu_list
        current_model = start_dag

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.

        best_operation = None
        best_score_delta = float("-inf")
        if custom_Q_table is None:
            # Select the best one in the Q_table
            for (s, a), q_value in self.Q_table.items():
                if s == current_model:
                    if q_value > best_score_delta:
                        best_score_delta = q_value
                        best_operation = a
        else:
            for (s, a), q_value in custom_Q_table.items():
                if s == current_model:
                    if q_value > best_score_delta:
                        best_score_delta = q_value
                        best_operation = a


        if best_operation == None or best_score_delta < 0 or (np.random.random() < self.e_epsilon):
            if self.greedy:
                best_operation, best_score_delta = max(
                    self._legal_operations(
                        current_model,
                        score_fn,
                        score.structure_prior_ratio,
                        max_indegree,
                        black_list,
                        white_list,
                        fixed_edges,
                    ),
                    key=lambda t: t[1],
                    default=(None, None),
                )
                if self.log:
                    print("Generative state is taking a Hill Climbing Step")
            else:
                # Epsilon: a random action in the available list
                action_list = []
                for i in self._legacy_legal_operations(
                        current_model,
                        score_fn,
                        score.structure_prior_ratio,
                        max_indegree,
                        black_list,
                        white_list,
                        fixed_edges,
                ):
                    action_list.append(i)
                if (len(action_list) > 0):
                    random_index = random.randint(0, len(action_list) - 1)
                    best_operation, best_score_delta = action_list[random_index]
                else:
                    best_operation, best_score_delta = (None, None)

                if self.log:
                    print("Generative state is taking a Random Step: ", best_operation, best_score_delta)
        else:
            if self.log:
                print("Generative state is taking a RL Step using experience")

        # print("best delta bic: ", best_score_delta,"which is: ", best_operation, "tabu: ",self.tabu_list)

        # if best_operation is None or best_score_delta < epsilon:
        if best_operation is None:
            return None  # None
        if best_operation[0] == "+":
            self.tabu_list.append(("-", best_operation[1]))

        return best_operation
