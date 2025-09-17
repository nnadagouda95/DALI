import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.stats import norm, cauchy
import pickle
import time
import sys
from enum import Enum
from matplotlib import pyplot as plt
from active_estimate_closed_form_posterior_single_response import ActiveEstimator, AdaptType, pair2hyperplane, normalized_kendall_tau_distance_sampled
import os, sys
import datetime


import multiprocessing
multiprocessing.set_start_method("fork")

"""
Script to run experiments where only one randomly chosen user is queried per iteration.
"""

exper_types = [['InfoGain', AdaptType.INFOGAIN],
               ['Uncertainty', AdaptType.UNCERTAINTY],
               ['Alternating-InfoGain', AdaptType.ALTERNATING_INFOGAIN],
               ['Alternating-Uncertainty', AdaptType.ALTERNATING_UNCERTAINTY],
               ['Random', AdaptType.RANDOM],
               ]

D = 10 # dimension
N = 1000 # number of items
M = 100 # number of queries
ntrials = 1 # number of trials
pair_samp_rate = 0.001 # candidate query sub sampling rate
k = 10 #noise constant
methods = ['Random']
# methods = ['Random', 'Alternating-InfoGain', 'Alternating-Uncertainty'] # methods to run

timeStamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
modelPath = './Result'
filePath = './Result/' + timeStamp
os.makedirs(filePath, exist_ok=True)
sys.stdout = open(filePath + '/run_info', 'w')

print('Querying one random user per iteration')
print('closed-form posterior update')
print('embedding dimension:', D)
print('num items:', N)
print('M:', M)
print('ntrials:', ntrials)
print('pair_samp_rate:', pair_samp_rate)
print('k:', k)
print(methods)

seeds = 2020 + np.arange(ntrials)
rand = seeds[0]
np.random.seed(rand)

bounds = np.array([-1, 1])
Embedding = np.random.uniform(4*bounds[0], 4*bounds[1], (N, D))
Embedding_mean = np.mean(Embedding, 0)
Embedding = Embedding - Embedding_mean.reshape((1, -1))

exper_types = [e for e in exper_types if e[0] in methods]
print('methods:', [e[0] for e in exper_types])
print('embedding: {} points x {} dimens'.format(N, D))


def main():

    out_data = {}
    for e in exper_types:
        out_data[e[0]] = {
            'user1': {'W_hist': [], 'W_sim': [], 'timer_vec': [], 'error_vec': [], 'kt_vec': []},
            'user2': {'W_hist': [], 'W_sim': [], 'timer_vec': [], 'error_vec': [], 'kt_vec': []}
        }

    for trial in range(ntrials):
        seed = seeds[trial]
        np.random.seed(seed)

        # Ground truth for two users
        mu_sim_pair = np.random.uniform(bounds[0], bounds[1], (2, D))
        cov_sim_pair = np.stack([np.identity(D), np.identity(D)])
        
        # Draw the true W vectors from the initial prior
        W_sim1 = np.random.multivariate_normal(mean=mu_sim_pair[0], cov=cov_sim_pair[0])
        W_sim2 = np.random.multivariate_normal(mean=mu_sim_pair[1], cov=cov_sim_pair[1])
        W_sim_pair = np.stack([W_sim1, W_sim2])

        for e in exper_types:
            print('trial', trial+1, '/', ntrials, 'experiment', e[0])

            (W_hist1, W_hist2), timer_vec = run_experiment(e[1], W_sim_pair, mu_sim_pair, cov_sim_pair, seed)
            
            # Process results for User 1
            error1 = np.linalg.norm(W_hist1 - W_sim1, axis=1) ** 2 / np.linalg.norm(W_sim1) ** 2
            kt_dist1 = normalized_kendall_tau_distance_sampled(W_hist1, W_sim1, Embedding)
            o1 = out_data[e[0]]['user1']
            o1['W_hist'].append(W_hist1); o1['W_sim'].append(W_sim1); o1['timer_vec'].append(timer_vec)
            o1['error_vec'].append(error1); o1['kt_vec'].append(kt_dist1)

            # Process results for User 2
            error2 = np.linalg.norm(W_hist2 - W_sim2, axis=1) ** 2 / np.linalg.norm(W_sim2) ** 2
            kt_dist2 = normalized_kendall_tau_distance_sampled(W_hist2, W_sim2, Embedding)
            o2 = out_data[e[0]]['user2']
            o2['W_hist'].append(W_hist2); o2['W_sim'].append(W_sim2); o2['timer_vec'].append(timer_vec)
            o2['error_vec'].append(error2); o2['kt_vec'].append(kt_dist2)
            
            print('\n')

    np.save(filePath + '/results_random_user.npy', out_data)
    
    # plotting the results
    x = range(0, M + 1)
    
    fig_mse_u1, ax_mse_u1 = plt.subplots(); fig_mse_u2, ax_mse_u2 = plt.subplots()
    fig_kt_u1, ax_kt_u1 = plt.subplots(); fig_kt_u2, ax_kt_u2 = plt.subplots()
    
    for e in exper_types:
        # User 1 Plots
        error_val1 = np.array(out_data[e[0]]['user1']['error_vec'])
        ax_mse_u1.plot(x, np.median(error_val1, axis=0), linewidth=4, label=e[0])
        ax_mse_u1.fill_between(x, np.quantile(error_val1, 0.25, axis=0), np.quantile(error_val1, 0.75, axis=0), alpha=0.2)
        kt_val1 = np.array(out_data[e[0]]['user1']['kt_vec'])
        ax_kt_u1.plot(x, np.median(kt_val1, axis=0), linewidth=4, label=e[0])
        ax_kt_u1.fill_between(x, np.quantile(kt_val1, 0.25, axis=0), np.quantile(kt_val1, 0.75, axis=0), alpha=0.2)
        
        # User 2 Plots
        error_val2 = np.array(out_data[e[0]]['user2']['error_vec'])
        ax_mse_u2.plot(x, np.median(error_val2, axis=0), linewidth=4, label=e[0])
        ax_mse_u2.fill_between(x, np.quantile(error_val2, 0.25, axis=0), np.quantile(error_val2, 0.75, axis=0), alpha=0.2)
        kt_val2 = np.array(out_data[e[0]]['user2']['kt_vec'])
        ax_kt_u2.plot(x, np.median(kt_val2, axis=0), linewidth=4, label=e[0])
        ax_kt_u2.fill_between(x, np.quantile(kt_val2, 0.25, axis=0), np.quantile(kt_val2, 0.75, axis=0), alpha=0.2)

    plot_configs = [
        {'ax': ax_mse_u1, 'fig': fig_mse_u1, 'ylabel': 'MSE', 'title': 'User 1 MSE', 'file': 'mse_plot_user1.png', 'yscale': 'log'},
        {'ax': ax_mse_u2, 'fig': fig_mse_u2, 'ylabel': 'MSE', 'title': 'User 2 MSE', 'file': 'mse_plot_user2.png', 'yscale': 'log'},
        {'ax': ax_kt_u1, 'fig': fig_kt_u1, 'ylabel': 'KT dist', 'title': 'User 1 KT Distance', 'file': 'kt_plot_user1.png'},
        {'ax': ax_kt_u2, 'fig': fig_kt_u2, 'ylabel': 'KT dist', 'title': 'User 2 KT Distance', 'file': 'kt_plot_user2.png'}
    ]

    for config in plot_configs:
        ax, fig = config['ax'], config['fig']
        ax.set_xlabel('Number of queries'); ax.set_ylabel(config['ylabel'])
        if 'yscale' in config: ax.set_yscale(config['yscale'])
        ax.set_title(f"{config['title']} | D: {D}, N: {N}, k: {k}")
        leg = ax.legend(prop={'size': 20}); leg.get_frame().set_facecolor('none'); leg.get_frame().set_linewidth(0.0)
        fig.set_size_inches(12, 9)
        fig.savefig(filePath + '/' + config['file'], dpi=100, bbox_inches='tight')
        

def run_experiment(adaptive, W_sim_pair, mu_sim_pair, cov_sim_pair, seed):
    # A single estimator manages both users, but updates them one by one
    estimator = ActiveEstimator()
    estimator.initialize(Embedding, k, adaptive, seed, modelPath, filePath, mu_sim_pair, cov_sim_pair, pair_samp_rate)
    
    def oracle(p, W_sim_user):
        (a, tau, b) = pair2hyperplane(p)
        num = k*(np.dot(a, W_sim_user) - tau)
        y = int(np.random.binomial(1, sp.special.expit(num)))
        return {'y': y}

    W_hist1 = np.zeros((M+1, D))
    W_hist2 = np.zeros((M+1, D))
    timer_vec = np.zeros(M+1)
    tic = time.time()

    # Initialize posteriors by sampling from the prior
    estimator.update_posterior(0)
    estimator.update_posterior(1)

    for i in range(M):
        # Store estimate before query i
        W_est1, W_est2 = estimator.getEstimates()
        W_hist1[i, :] = W_est1
        W_hist2[i, :] = W_est2
        timer_vec[i] = time.time() - tic
        
        print(f"Measurement {i+1} / {M}")

        # Select a single query based on current posteriors
        query = estimator.select_query()
        
        # Randomly choose which user to query
        user_to_query = np.random.randint(0, 2)
        print(f"--> Querying User {user_to_query + 1}")
        
        # Get response from ONLY the chosen user
        oracle_out = oracle(query, W_sim_pair[user_to_query])

        # Add observation for that user
        estimator.add_observation(query, oracle_out, user_to_query)

        # Update ONLY that user's posterior
        estimator.update_posterior(user_to_query)

    # Store the final estimate after all M queries
    W_est1, W_est2 = estimator.getEstimates()
    W_hist1[M, :] = W_est1
    W_hist2[M, :] = W_est2
    timer_vec[M] = time.time() - tic

    return (W_hist1, W_hist2), timer_vec

if __name__ == "__main__":
    main()