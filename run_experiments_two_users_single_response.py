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
import copy



"""
Script to run experiments where only one randomly chosen user is queried per iteration.
"""

exper_types = [['InfoGain', AdaptType.INFOGAIN],
               ['Uncertainty', AdaptType.UNCERTAINTY],
               ['Alternating-InfoGain', AdaptType.ALTERNATING_INFOGAIN],
               ['Alternating-Uncertainty', AdaptType.ALTERNATING_UNCERTAINTY],
               ['Random', AdaptType.RANDOM],
               ['MGuided-InfoGain', AdaptType.INFOGAIN_MIDPOINT],
               ]

nusers = 7
D = 10 # dimension
N = 1000 # number of items
M = 100 # number of queries
ntrials = 1 # number of trials
pair_samp_rate = 0.001 # candidate query sub sampling rate
k = 10 #noise constant
#methods = ['MGuided-InfoGain']
#methods = ['Random', 'Alternating-InfoGain', 'Alternating-Uncertainty'] # methods to run
methods = [
    'InfoGain',
    'Uncertainty',
    'Alternating-InfoGain',
    'Alternating-Uncertainty',
    'Random',
    'MGuided-InfoGain',
]


timeStamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
modelPath = './Result'
filePath = './Result/' + timeStamp
os.makedirs(filePath, exist_ok=True)
sys.stdout = open(filePath + '/run_info', 'w')

print('Querying one random user per iteration')
print('closed-form posterior update')
print('num users:', nusers)
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
            f'user{i}': {
                'W_hist': [],
                'W_sim': [],
                'timer_vec': [],
                'error_vec': [],
                'kt_vec': [],
            } for i in range(nusers)
        }
        out_data[e[0]]['G'] = copy.deepcopy(out_data[e[0]]['user0'])

    for trial in range(ntrials):
        seed = seeds[trial]
        np.random.seed(seed)

        # Ground truth for muliple users
        mu_sim_group = np.random.uniform(bounds[0], bounds[1], (nusers, D))
        cov_sim_group = np.stack([np.identity(D)]*nusers)

        # Draw the true W vectors from the initial prior
        W_sim_group = np.stack([
            np.random.multivariate_normal(mean=mu_sim, cov=cov_sim)
            for mu_sim, cov_sim in zip(mu_sim_group, cov_sim_group)
        ])
        G_sim = np.mean(W_sim_group, axis=0)

        for e in exper_types:
            print('trial', trial+1, '/', ntrials, 'experiment', e[0])

            W_hist_group, G_hist, timer_vec = run_experiment(e[1], W_sim_group, mu_sim_group, cov_sim_group, seed)

            # Process results for users
            for i, (W_sim, W_hist) in enumerate(zip(W_sim_group, W_hist_group)):
                error = np.linalg.norm(W_hist - W_sim, axis=1) ** 2 / np.linalg.norm(W_sim) ** 2
                kt_dist = normalized_kendall_tau_distance_sampled(W_hist, W_sim, Embedding)
                o = out_data[e[0]][f'user{i}']
                o['W_hist'].append(W_hist)
                o['W_sim'].append(W_sim)
                o['timer_vec'].append(timer_vec)
                o['error_vec'].append(error)
                o['kt_vec'].append(kt_dist)

            error = np.linalg.norm(G_hist - G_sim, axis=1) ** 2 / np.linalg.norm(G_sim) ** 2
            kt_dist = normalized_kendall_tau_distance_sampled(G_hist, G_sim, Embedding)
            o = out_data[e[0]]['G']
            o['W_hist'].append(W_hist)
            o['W_sim'].append(W_sim)
            o['timer_vec'].append(timer_vec)
            o['error_vec'].append(error)
            o['kt_vec'].append(kt_dist)

            print('\n')

    np.save(filePath + '/results_random_user.npy', out_data)

    # plotting the results
    x = range(0, M + 1)

    for i in range(nusers):
        fig_mse, ax_mse = plt.subplots()
        fig_kt, ax_kt = plt.subplots()

        for e in exper_types:
            # User plots
            o = out_data[e[0]][f'user{i}']
            error_val = np.array(o['error_vec'])
            ax_mse.plot(x, np.median(error_val, axis=0), linewidth=4, label=e[0])
            ax_mse.fill_between(x, np.quantile(error_val, 0.25, axis=0), np.quantile(error_val, 0.75, axis=0), alpha=0.2)
            kt_val = np.array(o['kt_vec'])
            ax_kt.plot(x, np.median(kt_val, axis=0), linewidth=4, label=e[0])
            ax_kt.fill_between(x, np.quantile(kt_val, 0.25, axis=0), np.quantile(kt_val, 0.75, axis=0), alpha=0.2)

        plot_configs = [
            {'ax': ax_mse, 'fig': fig_mse, 'ylabel': 'MSE', 'title': f'User {i} MSE', 'file': f'mse_plot_user{i}.png', 'yscale': 'log'},
            {'ax': ax_kt, 'fig': fig_kt, 'ylabel': 'KT dist', 'title': f'User {i} KT Distance', 'file': f'kt_plot_user{i}.png'},
        ]

        for config in plot_configs:
            ax, fig = config['ax'], config['fig']
            ax.set_xlabel('Number of queries'); ax.set_ylabel(config['ylabel'])
            if 'yscale' in config: ax.set_yscale(config['yscale'])
            ax.set_title(f"{config['title']} | D: {D}, N: {N}, k: {k}")
            leg = ax.legend(prop={'size': 20}); leg.get_frame().set_facecolor('none'); leg.get_frame().set_linewidth(0.0)
            fig.set_size_inches(12, 9)
            fig.savefig(filePath + '/' + config['file'], dpi=100, bbox_inches='tight')

    fig_mse, ax_mse = plt.subplots()
    fig_kt, ax_kt = plt.subplots()

    for e in exper_types:
        # User plots
        o = out_data[e[0]]['G']
        error_val = np.array(o['error_vec'])
        ax_mse.plot(x, np.median(error_val, axis=0), linewidth=4, label=e[0])
        ax_mse.fill_between(x, np.quantile(error_val, 0.25, axis=0), np.quantile(error_val, 0.75, axis=0), alpha=0.2)
        kt_val = np.array(o['kt_vec'])
        ax_kt.plot(x, np.median(kt_val, axis=0), linewidth=4, label=e[0])
        ax_kt.fill_between(x, np.quantile(kt_val, 0.25, axis=0), np.quantile(kt_val, 0.75, axis=0), alpha=0.2)

    plot_configs = [
        {'ax': ax_mse, 'fig': fig_mse, 'ylabel': 'MSE', 'title': f'Consensus (G) MSE', 'file': 'mse_plot_G.png', 'yscale': 'log'},
        {'ax': ax_kt, 'fig': fig_kt, 'ylabel': 'KT dist', 'title': f'Consensus (G) KT Distance', 'file': f'kt_plot_G.png'},
    ]

    for config in plot_configs:
        ax, fig = config['ax'], config['fig']
        ax.set_xlabel('Number of queries'); ax.set_ylabel(config['ylabel'])
        if 'yscale' in config: ax.set_yscale(config['yscale'])
        ax.set_title(f"{config['title']} | D: {D}, N: {N}, k: {k}")
        leg = ax.legend(prop={'size': 20}); leg.get_frame().set_facecolor('none'); leg.get_frame().set_linewidth(0.0)
        fig.set_size_inches(12, 9)
        fig.savefig(filePath + '/' + config['file'], dpi=100, bbox_inches='tight')


def run_experiment(adaptive, W_sim_group, mu_sim_group, cov_sim_group, seed):
    # A single estimator manages both users, but updates them one by one
    estimator = ActiveEstimator()
    estimator.initialize(Embedding, k, adaptive, seed, modelPath, filePath, mu_sim_group, cov_sim_group, pair_samp_rate)

    def oracle(p, W_sim):
        (a, tau, b) = pair2hyperplane(p)
        num = k*(np.dot(a, W_sim) - tau)
        y = int(np.random.binomial(1, sp.special.expit(num)))
        return {'y': y}

    W_hist_group = [np.zeros((M+1, D)) for _ in range(nusers)]
    G_hist = np.zeros((M+1, D))
    timer_vec = np.zeros(M+1)
    tic = time.time()

    # Initialize posteriors by sampling from the prior
    for i in range(nusers):
        estimator.update_posterior(i)

    for j in range(M):
        # Store estimate before query j
        W_est_group, G_est = estimator.getEstimates()
        for W_hist, W_est in zip(W_hist_group, W_est_group):
            W_hist[j, :] = W_est
        G_hist[j, :] = G_est
        timer_vec[j] = time.time() - tic

        print(f"Measurement {j+1} / {M}")

        # Select a single query based on current posteriors
        query = estimator.select_query()

        # Randomly choose which user to query
        user_to_query = np.random.randint(0, nusers)
        print(f"--> Querying User {user_to_query + 1}")

        # Get response from ONLY the chosen user
        oracle_out = oracle(query, W_sim_group[user_to_query])

        # Add observation for that user
        estimator.add_observation(query, oracle_out, user_to_query)

        # Update ONLY that user's posterior
        estimator.update_posterior(user_to_query)

    # Store the final estimate after all M queries
    W_est_group, G_est = estimator.getEstimates()
    for W_hist, W_est in zip(W_hist_group, W_est_group):
        W_hist[M, :] = W_est
    G_hist[M, :] = G_est
    timer_vec[M] = time.time() - tic

    return W_hist_group, G_hist, timer_vec

if __name__ == "__main__":
    main()