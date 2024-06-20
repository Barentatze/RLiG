from copy import deepcopy

from pgmpy.metrics import log_likelihood_score

from ..kdb import *
from ..kdb import _add_uniform
from ..utils import *
from ..structure_learning.HillClimbing import HillClimbSearch
from ..structure_learning.utils.buffer import StackBuffer
from ..structure_learning.RL_agent import ReinforcementLearningAgent
from ..structure_learning.K2_agent import K2Agent
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
import tensorflow as tf

class RLiG:
    """
    The RLiG Model.
    """
    def __init__(self) -> None:
        self._d = None
        self.__gen_weights = None
        self.batch_size = None
        self.epochs = None
        self.k = None
        self.constraints = None
        self._ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1) #将分类特征转换成整数，未知时编码为-1
        self._label_encoder   = LabelEncoder() #一般用于目标变量的编码，例如分类问题中的目标标签
    
    def fit(self, x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=1, gan=1, n = 4):
        '''
        Fit the model to the given data.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete.

        y : array_like of shape (n_samples,)
            Label of the dataset.

        k : int, default=0
            Parameter k of RLiG model. Must be greater than 0. No more than 2 is Suggested.

        batch_size : int, default=32
            Size of the batch to feed the model at each step.

        epochs : int, default=0
            Number of epochs to use during training.

        warmup_epochs : int, default=1
            Number of epochs to use in warmup phase. Defaults to :attr:`1`.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        gan: bool, default=1
            Whether to use the discriminator to implement a gan structure. Use 1 for enabling the
            discriminator and GAN Structure

        n: int, default=4
            The definition of the n-steps in the n-SARSA

        Returns
        -------
        self : object
            Fitted model.
        '''

        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        self.variables = list(x.columns.values)
        x_int = self._ordinal_encoder.fit_transform(x)
        y_int = self._label_encoder.fit_transform(y).astype(int)
        #Dataframe -> Numpy.ndarry
        d = DataUtils(x_int, y_int)
        self._d = d # DataUtils
        self.k = k # k for kdb
        self.batch_size = batch_size

        score_buffer = StackBuffer()
        hc_agent = HillClimbSearch(data=x)
        rl_agent = K2Agent(data_X= x, data_Y= y)

        if verbose:
            print(f"warmup run:")
        history = self._warmup_run(warmup_epochs, verbose=verbose) #在warmup run中创建了 KDB

        # Init the Bayesian Network using the Hill Climbing Agent
        self.bayesian_network, best_action = hc_agent.estimate_once() #As this is the init, no need for parameter
        self.bayesian_network.fit(x) #BayesianNetwork.fit(values)
        original_score = log_likelihood_score(self.bayesian_network, x) #loglikelihood(model,data) 可能不是x,可能要是整个data
        #这里要看看实现？关于log_likelihood_score决定是否要使用x还是整个dataset

        for _ in range(epochs):
            for i in range(n):
                # Take a HillClimbing Step
                current_structure, action = hc_agent.estimate_once(start_dag=self.bayesian_network) # current_structure: Bayesian Network
                # Calculate Score and Reward
                current_structure.fit(x)
                current_score = log_likelihood_score(current_structure, x)
                reward = current_score - original_score
                # Buffer it
                score_buffer.push((self.bayesian_network, action, reward,
                                   current_structure))  # stackbuffer(S,A,R,S')
                # Update the Step
                self.bayesian_network = current_structure
                original_score = current_score

            # Do a Reinforcement Learning
            # 这里Reinforcement Learning只是take_action, 具体的reward实现并不在这里

            # Take a reinforcment learning step
            # self.bayesian_network, action = rl_agent.estimate_once(
            #     start_dag=self.bayesian_network) # Use Reinforcement Learning agent to take a step
            # # fit the baysian structure, as for the generator
            # self.bayesian_network.fit(x)

            # Do a GAN
            # syn_data = self._sample(verbose=0)# The original sampling method in the ganblr
            data_sampler = BayesianModelSampling(
                self.bayesian_network)  # Parameters: model (instance of BayesianNetwork) – model on which inference queries will be computed
            syn_data = data_sampler.forward_sample(size=5000)

            discriminator_label = np.hstack([np.ones(d.data_size), np.zeros(d.data_size)])  # 长度为2N的数组，前一半是1后一半是0
            # Train the GAN Structure.
            if (gan == 0):  # Using Gan
                discriminator_input = np.vstack([x_int, syn_data[:, :-1]])
                disc_input, disc_label = sample(discriminator_input, discriminator_label, frac=0.8)
                disc = self._discrim()
                d_history = disc.fit(disc_input, disc_label, batch_size=batch_size, epochs=1,
                                     verbose=0).history  # discriminator fit
                prob_fake = disc.predict(x, verbose=0)  # fake的概率，要看disc的实现 因为我改了x的定义这句话可能是错的
                ls = np.mean(-np.log(np.subtract(1, prob_fake)))  # 1-prob_fake reward中的第二项
            else:
                ls = np.mean(-np.log(1))

            # g_history = self._run_generator(loss=ls).history
            # syn_data = self._sample(verbose=0) #sample syn data
            # syn_data = data_sampler.forward_sample(size=5000)

            # if verbose:
            #     if (gan == 1):
            #         print(
            #             f"Epoch {i + 1}/{epochs}: G_loss = {g_history['loss'][0]:.6f}, G_accuracy = {g_history['accuracy'][0]:.6f}, D_loss = {d_history['loss'][0]:.6f}, D_accuracy = {d_history['accuracy'][0]:.6f}")
            #     else:
            #         print(
            #             f"Epoch {i + 1}/{epochs}: G_loss = {g_history['loss'][0]:.6f}, G_accuracy = {g_history['accuracy'][0]:.6f}")

            current_score = log_likelihood_score(self.bayesian_network, x)
            reward = current_score - original_score
            original_score = current_score

            # Update the Q value using stack buffer

            # StackBuffer -> Update -> RL Q-Table Buffer -> Sampling -> Learn
            # Bayesian -> DAG? Remove the CPD information?

            # Memorize the Q experience into the table


        return self
        
    def evaluate(self, x, y, model='lr') -> float:
        """
        Perform a TSTR(Training on Synthetic data, Testing on Real data) evaluation.

        Parameters
        ----------
        x, y : array_like
            Test dataset.

        model : str or object
            The model used for evaluate. Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.
            几种模型直接就是实现好的，看这里
        Return:
        --------
        accuracy_score : float.

        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score
        
        eval_model = None
        models = dict(
            lr=LogisticRegression,
            rf=RandomForestClassifier,
            mlp=MLPClassifier
        )
        if model in models.keys():
            eval_model = models[model]()
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):
            eval_model = model
        else:
            raise Exception("Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.")

        synthetic_data = self._sample() #Sample the synthetic data
        synthetic_x, synthetic_y = synthetic_data[:,:-1], synthetic_data[:,-1]
        x_test = self._ordinal_encoder.transform(x) #The real dataset
        y_test = self._label_encoder.transform(y)

        #Testing the model
        categories = self._d.get_categories()
        pipline = Pipeline([('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore')), ('model',  eval_model)]) 
        pipline.fit(synthetic_x, synthetic_y) #TS
        pred = pipline.predict(x_test) #TR
        return accuracy_score(y_test, pred)
    
    def sample(self, size=None, verbose=1) -> np.ndarray:
        """
        Generate synthetic data.     

        Parameters
        ----------
        size : int or None
            Size of the data to be generated. set to `None` to make the size equal to the size of the training set.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.
        
        Return:
        -----------------
        synthetic_samples : np.ndarray
            Generated synthetic data.
        """
        ordinal_data = self._sample(size, verbose)
        origin_x = self._ordinal_encoder.inverse_transform(ordinal_data[:,:-1]) #因为在_sample中是按照序数编码格式，所以这一步是将序数编码格式转换为原格式
        origin_y = self._label_encoder.inverse_transform(ordinal_data[:,-1]).reshape(-1,1)
        return np.hstack([origin_x, origin_y])
        
    def _sample(self, size=None, verbose=1) -> np.ndarray:
        #The sample need to be modified
        """
        Generate synthetic data in ordinal encoding format 按照序数编码格式
        """

        """
        每个节点表示一个随机变量，并且每个节点都有一个 CPD 表，用来表示该节点在其父节点给定的条件下的概率分布。具体来说，如果我们有一个节点 X
        X 和它的父节点 Parents(X)，那么 CPD 表表示 P(X∣Parents(X))。
        
        | A | B | P(C=1 | A, B) | P(C=0 | A, B) |
        |---|---|--------------|--------------|
        | 0 | 0 | 0.1 | 0.9 |
        | 0 | 1 | 0.4 | 0.6 |
        | 1 | 0 | 0.7 | 0.3 |
        | 1 | 1 | 0.9 | 0.1 |
        """
        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        #basic varibles
        d = self._d #DataUtil 应该使用数据生成的，所以要看一下DataUtil是什么
        feature_cards = np.array(d.feature_uniques)#特征的所有取值
        #ensure sum of each constraint group equals to 1, then re concat the probs
        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist()) #这个应该是在生成所有的约束
        constraint_idxs = [(_idxs[i],_idxs[i+1]) for i in range(len(_idxs)-1)]
        
        #生成CPD表
        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end,:] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p/p.sum(axis=0) for p in cpd_probs])

        #assign the probs to the full cpd tables
        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
        have_value_idxs = d._kdbe.have_value_idxs_
        full_cpd_probs = [] 
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            #(n_high_order_feature_uniques, n_classes)
            cpd_prob_ = cpd_probs[start:end,:]
            #(n_all_combination) Note: the order is (*parent, variable)
            have_value_ravel = have_value.ravel()
            #(n_classes * n_all_combination)
            have_value_ravel_repeat = np.hstack([have_value_ravel] * d.num_classes)
            #(n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
            full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            #(n_classes * n_parent_combinations, n_variable_unique)
            full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
            full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
            full_cpd_probs.append(full_cpd_prob)
    
        #prepare node and edge names
        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i,j in d._kdbe.edges_]
        y_name = node_names[-1]
    
        #create TabularCPD objects
        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(str(name), feature_cards[name], table, 
                       evidence=[y_name, *[str(e) for e in evidences]], 
                       evidence_card=[d.num_classes, *feature_cards[evidences].tolist()])
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (d.class_counts/d.data_size).reshape(-1,1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)
    
        #create kDB model, then sample the data
        model = BayesianNetwork(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        sample_size = d.data_size if size is None else size
        result = BayesianModelSampling(model).forward_sample(size=sample_size, show_progress = verbose > 0)
        sorted_result = result[node_names].values
        
        return sorted_result
    
    def _warmup_run(self, epochs, verbose=None):
        d = self._d
        tf.keras.backend.clear_session()
        ohex = d.get_kdbe_x(self.k)
        self.constraints = softmax_weight(d.constraint_positions)
        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        history = elr.fit(ohex, d.y, batch_size=self.batch_size, epochs=epochs, verbose=verbose)
        self.__gen_weights = elr.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _run_generator(self, loss):
        d = self._d
        ohex = d.get_kdbe_x(self.k)#获得一个高阶的？ Higher-order feature
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(d.num_classes, input_dim=ohex.shape[1], activation='softmax',kernel_constraint=self.constraints))
        model.compile(loss=elr_loss(loss), optimizer='adam', metrics=['accuracy'])
        model.set_weights(self.__gen_weights)
        history = model.fit(ohex, d.y, batch_size=self.batch_size,epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()
        tf.keras.backend.clear_session()
        return history
    
    def _discrim(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1, input_dim=self._d.num_features, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model