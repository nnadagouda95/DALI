import numpy as np
import scipy.special as sc
import scipy.stats as st
import scipy as sp
from scipy.optimize import minimize
import pystan
# import stan
import pickle
from enum import Enum
# import arviz as az
from matplotlib import pyplot as plt
from scipy.stats import cauchy
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar


"""
Implementation of query selection methods for two users within a single estimator.
"""


class ActiveEstimator():

    """
    Search object for query selection methods, managing two users simultaneously.
    """

    my_model = """
    data {
        int<lower=0> D;         // space dimension
        int<lower=0> M;         // number of measurements so far
        vector[2] bounds;       // hypercube bounds [lower,upper]
        real<lower=0> k;        // noise constant
        int y[M];               // measurement outcomes
        vector[D] A[M];         // hyperplane directions
        vector[M] tau;          // hyperplane offsets
        vector[D] B[M];         // mid points
    }
    parameters {
        vector<lower=bounds[1],upper=bounds[2]>[D] W;         // the user point
    }
    transformed parameters {
        vector[M] z;
        for (i in 1:M){
                real num  = k*(dot_product(A[i], W) - tau[i]);
                z[i] = num;
            }
    }
    model {
        // prior
        W ~ uniform(bounds[1],bounds[2]);

        // linking observations
        y ~ bernoulli_logit(z);
    }
    """


    def __init__(self):
        # pystan version
        try:
            self.sm = pickle.load(open('./Result/model.pkl', 'rb'))
        except:
            self.sm = pystan.StanModel(model_code=self.my_model)
            with open('./Result/model.pkl', 'wb') as f:
                pickle.dump(self.sm, f)
                

    
    def initialize(self, embedding, k, method, seed, model_path,
                   path, bounds=np.array([-1, 1]), pair_samp_rate=0.01, 
                   Nchains=4, Nsamples=4000, diagnostic = False):
        """
        Initializes the search object for two users.
        """
        self.embedding = embedding
        self.N = embedding.shape[0]
        self.D = embedding.shape[1]

        self.k = k
        self.method = method
        self.seed = seed
        self.model_path = model_path
        self.path = path
        self.bounds = bounds
        self.Npairs = int(pair_samp_rate * sp.special.comb(self.N, 2))

        self.Nchains = Nchains
        self.Nsamples = Nsamples
        Niter = int(2*Nsamples/Nchains)
        assert Niter >= 1000
        self.Niter = Niter

        # State variables are now lists of size 2 (one for each user)
        self.mean_W = [np.zeros(self.D) for _ in range(2)]
        self.cov_W = [np.zeros((self.D, self.D)) for _ in range(2)]
        self.W_samples = [None, None]

        self.A = [[], []]
        self.tau = [[], []]
        self.B = [[], []]
        self.y_vec = [[], []]
        
        self.oracle_queries_made = []
        self.diagnostic = diagnostic
        np.random.seed(self.seed)

    def update_posterior(self):
        """
        Samples from the posterior for both users to update their belief states.
        """
        for user_idx in range(2):
            if not self.A[user_idx]:
                W_samples = np.random.uniform(
                    self.bounds[0], self.bounds[1], (self.Nsamples, self.D))
            else:
                data_gen = {'D': self.D, 'M': len(self.A[user_idx]),
                            'bounds': self.bounds,
                            'k': self.k,
                            'y': self.y_vec[user_idx],
                            'A': self.A[user_idx],
                            'tau': self.tau[user_idx],
                            'B': self.B[user_idx]}
                
                init_vals = {'W': self.mean_W[user_idx]}
                fit = self.sm.sampling(data=data_gen, iter=self.Niter, 
                                       chains=self.Nchains, init=[init_vals]*self.Nchains,
                                       seed = self.seed)
#                 print(f"User {user_idx+1} Posterior Fit:")
#                 print(fit)
                W_samples = fit.extract()['W']

            if W_samples.ndim < 2:
                W_samples = W_samples[:, np.newaxis]

            assert W_samples.shape == (self.Nsamples, self.D)
            
            self.mean_W[user_idx] = np.mean(W_samples, 0)
            self.cov_W[user_idx] = np.cov(W_samples, rowvar=False)
            self.W_samples[user_idx] = W_samples

    def select_query(self):
        """
        Selects a single query for both users, considering their joint posteriors.
        """
        if self.method == AdaptType.INFOGAIN:
            Pairs = self.get_random_pairs(self.N, self.Npairs)
            value = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                ind = Pairs[j]
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                # Sum of information gain for both users
                _, value1 = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[0])
                _, value2 = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[1])
                value[j] = value1 + value2

            ind = Pairs[np.argmax(value)]
            p = self.embedding[ind,:]
        
        
        elif self.method == AdaptType.UNCERTAINTY:
            Pairs = self.get_random_pairs(self.N, self.Npairs)
            value = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                ind = Pairs[j]
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                # Sum of uncertainty for both users
                value1, _  = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[0])
                value2, _ = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[1])
                value[j] = value1 + value2

            ind = Pairs[np.argmax(value)]
            p = self.embedding[ind,:]
        
        
        elif self.method == AdaptType.ALTERNATING_INFOGAIN:
            # Determine which user to optimize for based on the query number
            query_number = len(self.oracle_queries_made)
            user_to_optimize = query_number % 2  # Alternates between 0 and 1

            print(f"Query #{query_number + 1}: Optimizing for User {user_to_optimize + 1}")
            
            Pairs = self.get_random_pairs(self.N, self.Npairs)
            value = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                ind = Pairs[j]
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                
                # Calculate infogain ONLY for the chosen user for this turn
                _, value[j] = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[user_to_optimize])
        
        
            ind = Pairs[np.argmax(value)]
            p = self.embedding[ind,:]
        
        
        elif self.method == AdaptType.ALTERNATING_UNCERTAINTY:
            # Determine which user to optimize for based on the query number
            query_number = len(self.oracle_queries_made)
            user_to_optimize = query_number % 2  # Alternates between 0 and 1

            print(f"Query #{query_number + 1}: Optimizing for User {user_to_optimize + 1}")
            
            Pairs = self.get_random_pairs(self.N, self.Npairs)
            value = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                ind = Pairs[j]
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                
                # Calculate uncertainty ONLY for the chosen user for this turn
                value[j], _ = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[user_to_optimize])
                
            ind = Pairs[np.argmax(value)]
            p = self.embedding[ind,:]
        
        
        else:   # random pair method
            ind = np.random.choice(self.N, 2, replace=False)
            p = self.embedding[ind,:]
        
        return p
    

    def add_observation(self, p, oracle_out1, oracle_out2):
        """
        Updates the model with observations from both users for a single query.
        """
        (A_sel, tau_sel, B_sel) = pair2hyperplane(p)
        
        # Add data for User 1
        self.A[0].append(A_sel)
        self.tau[0].append(tau_sel)
        self.B[0].append(B_sel)
        self.y_vec[0].append(oracle_out1['y'])
        
        # Add data for User 2
        self.A[1].append(A_sel)
        self.tau[1].append(tau_sel)
        self.B[1].append(B_sel)
        self.y_vec[1].append(oracle_out2['y'])

        self.oracle_queries_made.append(p)
    
    
    def getEstimates(self):
        """
        Returns estimates of user points for both users.
        """
        return self.mean_W[0], self.mean_W[1]


    def evaluate_pair(self, a, tau, b, W_samples):
        # estimates mutual information of input pair
        _, Lik = self.likelihood_vec(a, tau, b, W_samples)
        Ftilde = np.mean(Lik)
        mutual_info = self.binary_entropy(Ftilde) - np.mean(
            self.binary_entropy(Lik))
        entropy = self.binary_entropy(Ftilde)
        return entropy, mutual_info
    

    def likelihood_vec(self, a, tau, b, W):
        # mutual information support function
        num = self.k*(np.dot(W,a) - tau)
        z = num
        return z, sp.special.expit(z)
        

    def binary_entropy(self, x):
        # mutual information support function
        return -(sc.xlogy(x, x) + sc.xlog1py(1 - x, -x))/np.log(2)

    def get_random_pairs(self, N, M):
        # pair selection support function
        indices = np.random.choice(N, (int(1.5*M), 2))
        indices = [(int(i[0]), int(i[1])) for i in indices if i[0] != i[1]]
        assert len(indices) >= M
        return indices[0:M]
    

class AdaptType(Enum):
    RANDOM = 0
    INFOGAIN = 1
    UNCERTAINTY = 2
    ALTERNATING_INFOGAIN = 3
    ALTERNATING_UNCERTAINTY = 4
    


def pair2hyperplane(p):
    # converts pair to hyperplane weights, bias and the mid point
    A_emb = 2*(p[0, :] - p[1, :])
    tau_emb = (np.linalg.norm(p[0, :])**2 - np.linalg.norm(p[1, :])**2)
    B_emb = (p[0, :] + p[1, :])/2
    return (A_emb, tau_emb, B_emb)


def normalized_kendall_tau_distance_sampled(W_hist, W_sim, embedding, batch_size=15, num_batches=100):
    """Compute the Kendall tau distance."""
    N = embedding.shape[0]
    W_sim = W_sim[None, :]
    true_distances = sp.spatial.distance.cdist(W_sim, embedding).squeeze()
    kt_dist = np.zeros(len(W_hist))

    for k in range(len(W_hist)):
        W_est = W_hist[k,:]
        W_est = W_est[None, :]
        estimated_distances = sp.spatial.distance.cdist(W_est, embedding).squeeze()
        n = len(true_distances)
        assert len(estimated_distances) == n

        kt_dist_est = 0
        for m in range(num_batches):
            sampled_indices = np.random.choice(N, batch_size, replace=False)
            sampled_true_distances = true_distances[sampled_indices]
            sampled_estimated_distances = estimated_distances[sampled_indices]

            i, j = np.meshgrid(np.arange(batch_size), np.arange(batch_size))
            a = np.argsort(sampled_true_distances)
            b = np.argsort(sampled_estimated_distances)
            num_disordered = (np.maximum(-np.sign((a[i] - a[j]) * (b[i] - b[j])), 0) * (i<j)).sum()
            kt_dist_est += (2*num_disordered) / (batch_size * (batch_size - 1))

        kt_dist[k] = kt_dist_est / num_batches
    return kt_dist