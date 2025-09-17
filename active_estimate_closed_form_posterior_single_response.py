import numpy as np
import scipy.special as sc
import scipy.stats as st
import scipy as sp
from scipy.optimize import minimize
# import pystan
# import stan
import pickle
from enum import Enum
# import arviz as az
from matplotlib import pyplot as plt
from scipy.stats import cauchy
from numpy.linalg import inv, norm
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar


"""
Implementation of query selection methods for two users, where only one is updated at a time.
"""


def update_given_y_logistic(k, p, q, y, sigma, mu):
    """
    UPDATE_GIVEN_Y_LOGISTIC Updates the posterior distribution of the parameters
    in a logistic regression model given the word's label.
    It follows the method by T. S. Jaakkola and M. I. Jordan,
    "Bayesian parameter estimation via variational methods,"
    Statistics and Computing, vol. 10, pp. 25–37, 2000.

    Input arguments:
        p      - numpy array (d,)
        q      - numpy array (d,)
        y      - Label is -1 or 1.
        sigma  - Prior covariance of the classifier (d x d).
        mu     - Prior mean of the classifier (d,).

    Returns:
        mu_pos     - Posterior mean of the classifier.
        sigma_pos  - Posterior covariance of the classifier.
    """
    # Ensure arrays
    p = np.asarray(p).reshape(-1)
    q = np.asarray(q).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma)

    x = k * 2.0 * (p - q)
    c = k * (norm(q)**2 - norm(p)**2)
    sigma_inv = inv(sigma)
    mu_prior = mu.copy()
    mu_past = mu_prior.copy()

    # xi initialization
    xi = np.sqrt(max(0.0, x @ sigma @ x + (x @ mu)**2 + 2.0 * c * (x @ mu) + c**2))

    difference = 1.0
    iteration_count = 0

    # Fixed-point iterations
    while difference > 1e-7:
        # Compute posterior
        # lambda = tanh(xi/2) / (4 * xi)
        # Guard against xi ~ 0 to avoid division by zero (keeping behavior close to original)
        if xi <= 1e-16:
            lam = 1.0 / 8.0
        else:
            lam = np.tanh(xi / 2.0) / (4.0 * xi)

        sigma_pos = inv(sigma_inv + 2.0 * lam * np.outer(x, x))
        mu_pos = sigma_pos @ (sigma_inv @ mu_prior + ((y / 2.0) - 2.0 * lam * c) * x)

        xi = max(1e-16, np.sqrt(x @ sigma_pos @ x + (x @ mu_pos)**2 + 2.0 * c * (x @ mu_pos) + c**2))

        # Check for stopping condition
        difference = norm(mu_pos - mu_past)
        mu_past = mu_pos.copy()
        iteration_count += 1
        if iteration_count > 1000:
            break

    return mu_pos, sigma_pos


class ActiveEstimator():

    """
    Search object managing two users, updating one at a time.
    """


    def __init__(self):
        pass

    
    def initialize(self, embedding, k, method, seed, model_path,
                   path, init_mean_W, init_cov_W, pair_samp_rate=0.01, 
                   diagnostic=False):
        """
        Initializes the search object for two users.
        """
        self.embedding = embedding
        self.N = embedding.shape[0]
        self.D = embedding.shape[1]

        self.k = k
        self.method = method
        self.seed = seed
        self.path = path
        self.Npairs = int(pair_samp_rate * sp.special.comb(self.N, 2))

        # Hardcoded sample count for posterior sampling
        self.Nsamples = 4000

        # State variables are lists of size 2 (one for each user)
        self.mean_W = [init_mean_W[0].copy(), init_mean_W[1].copy()]
        self.cov_W = [init_cov_W[0].copy(), init_cov_W[1].copy()]
        self.W_samples = [None, None]

        # User-specific observation data
        self.y_vec = [[], []]
        self.queries_for_user = [[], []]
        
        # Global list of all queries made
        self.oracle_queries_made = []
        self.diagnostic = diagnostic
        np.random.seed(self.seed)

    def update_posterior(self, user_index):
        """
        Updates the posterior for a single, specified user and resamples.
        """
        # Perform variational update if there's data for this user
        if self.queries_for_user[user_index]:
            p, q = self.queries_for_user[user_index][-1]
            y = self.y_vec[user_index][-1]
            
            # The label 'y' from the oracle is {0, 1}. Convert to {-1, 1} for the update rule.
            y_update = 1.0 if y == 1 else -1.0
            
            self.mean_W[user_index], self.cov_W[user_index] = update_given_y_logistic(
                self.k, p, q, y_update, self.cov_W[user_index], self.mean_W[user_index]
            )
        
        # Sample from the updated (or prior) Gaussian posterior
        try:
            W_samples = np.random.multivariate_normal(
            self.mean_W[user_index], self.cov_W[user_index], size=self.Nsamples)
        except np.linalg.LinAlgError:
            # If covariance is not positive semi-definite, add jitter
            jitter = np.eye(self.D) * 1e-6
            W_samples = np.random.multivariate_normal(
                self.mean_W[user_index], self.cov_W[user_index] + jitter, size=self.Nsamples)
        
        self.W_samples[user_index] = W_samples

        self.mean_G = (self.mean_W[0] + self.mean_W[1])/2
        self.cov_G = (self.cov_W[0] + self.cov_W[1])/4
        
        try:
            G_samples = np.random.multivariate_normal(
            self.mean_G, self.cov_G, size=self.Nsamples)
        except np.linalg.LinAlgError:
            # If covariance is not positive semi-definite, add jitter
            jitter = np.eye(self.D) * 1e-6
            G_samples = np.random.multivariate_normal(
                self.mean_G, self.cov_G + jitter, size=self.Nsamples)
        self.G_samples = G_samples

    def select_query(self):
        """
        Selects a single query for both users, considering their joint posteriors.
        """
        # Get a list of candidate pairs
        print('N:', self.N)
        print('Npairs/M:', self.Npairs)
        Pairs = self.get_random_pairs(self.N, self.Npairs)
        value = np.zeros((self.Npairs,))
        
        if 'Alternating' in self.method.name:
            query_number = len(self.oracle_queries_made)
            user_to_optimize = query_number % 2  # Alternates between 0 and 1
            print(f"Selecting query based on User {user_to_optimize + 1}'s posterior")
            
            for j, ind in enumerate(Pairs):
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                entropy, mutual_info = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[user_to_optimize])
                if 'InfoGain' in self.method.name:
                    value[j] = mutual_info
                else: # Uncertainty
                    value[j] = entropy

        elif self.method in [AdaptType.INFOGAIN, AdaptType.UNCERTAINTY]:
             for j, ind in enumerate(Pairs):
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                # Sum of metric for both users
                e1, mi1 = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[0])
                e2, mi2 = self.evaluate_pair(A_emb, tau_emb, B_emb, self.W_samples[1])
                if self.method == AdaptType.INFOGAIN:
                    value[j] = mi1 + mi2
                else: # Uncertainty
                    value[j] = e1 + e2
        
        elif self.method == AdaptType.INFOGAIN_MIDPOINT:
            for j, ind in enumerate(Pairs):
                p = self.embedding[ind,:]
                (A_emb, tau_emb, B_emb) = pair2hyperplane(p)
                _, value[j] = self.evaluate_pair(A_emb, tau_emb, B_emb, self.G_samples)


        else: # Random
            ind = np.random.choice(len(Pairs))
            return self.embedding[Pairs[ind],:]
        
        # Select the best pair based on the calculated values
        best_ind = Pairs[np.argmax(value)]
        return self.embedding[best_ind,:]
    

    def add_observation(self, p, oracle_out, user_index):
        """
        Adds an observation for a SINGLE specified user.
        """
        self.y_vec[user_index].append(oracle_out['y'])
        self.queries_for_user[user_index].append(p)
        self.oracle_queries_made.append(p)
    
    
    def getEstimates(self):
        """
        Returns estimates of user points for both users.
        """
        return self.mean_W[0], self.mean_W[1]


    def evaluate_pair(self, a, tau, b, W_samples):
        # estimates mutual information and entropy of input pair
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
    INFOGAIN_MIDPOINT = 5
    


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