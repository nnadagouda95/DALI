import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.stats import norm, cauchy
import pickle
import time
import sys
from enum import Enum
from matplotlib import pyplot as plt
from active_estimate_closed_form_posterior import ActiveEstimator, AdaptType, pair2hyperplane, normalized_kendall_tau_distance_sampled
import os, sys
import datetime


import multiprocessing
multiprocessing.set_start_method("fork")

"""
Script to run experiments for learning two users simultaneously with a single estimator.
"""

exper_types = [['InfoGain', AdaptType.INFOGAIN],
               ['Uncertainty', AdaptType.UNCERTAINTY],
               ['Alternating-InfoGain', AdaptType.INFOGAIN],
               ['Alternating-Uncertainty', AdaptType.UNCERTAINTY],
               ['Random', AdaptType.RANDOM],
               ]

D = 2 # dimension
N = 1000 # number of items
M = 100 # number of queries
ntrials = 1 # number of trials
pair_samp_rate = 0.001 # candidate query sub sampling rate
k = 10 #noise constant
methods = ['InfoGain', 'Uncertainty', 'Alternating-InfoGain', 'Alternating-Uncertainty', 'Random'] # methods

timeStamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
modelPath = './Result'
filePath = './Result/' + timeStamp
os.makedirs(filePath, exist_ok=True)
sys.stdout = open(filePath + '/run_info', 'w')

print('Simultaneously learning two users with a single estimator')
print('logistic model')
print('embedding dimension:', D)
print('num items:', N)
print('M:', M)
print('ntrials:', ntrials)
print('pair_samp_rate:', pair_samp_rate)
print('k:', k)
print(methods)
for m in methods:
    print('method: {}'.format(m))

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
        W_sim_pair = np.random.uniform(bounds[0], bounds[1], (2, D))

        for e in exper_types:
            print('trial', trial+1, '/', ntrials, 'experiment', e[0])

            (W_hist1, W_hist2), timer_vec = run_experiment(e[1], W_sim_pair, seed)
            
            W_sim1, W_sim2 = W_sim_pair[0,:], W_sim_pair[1,:]

            # Process results for User 1
            error1 = np.linalg.norm(W_hist1 - W_sim1, axis=1) ** 2 / np.linalg.norm(W_sim1) ** 2
            kt_dist1 = normalized_kendall_tau_distance_sampled(W_hist1, W_sim1, Embedding)
            o1 = out_data[e[0]]['user1']
            o1['W_hist'].append(W_hist1)
            o1['W_sim'].append(W_sim1)
            o1['timer_vec'].append(timer_vec)
            o1['error_vec'].append(error1)
            o1['kt_vec'].append(kt_dist1)

            # Process results for User 2
            error2 = np.linalg.norm(W_hist2 - W_sim2, axis=1) ** 2 / np.linalg.norm(W_sim2) ** 2
            kt_dist2 = normalized_kendall_tau_distance_sampled(W_hist2, W_sim2, Embedding)
            o2 = out_data[e[0]]['user2']
            o2['W_hist'].append(W_hist2)
            o2['W_sim'].append(W_sim2)
            o2['timer_vec'].append(timer_vec)
            o2['error_vec'].append(error2)
            o2['kt_vec'].append(kt_dist2)
            
            print('\n')

    np.save(filePath + '/results.npy', out_data)
    
    # plotting the results
    x = range(0, M + 1)
    
    # Create four separate figures and axes
    fig_mse_u1, ax_mse_u1 = plt.subplots()
    fig_mse_u2, ax_mse_u2 = plt.subplots()
    fig_kt_u1, ax_kt_u1 = plt.subplots()
    fig_kt_u2, ax_kt_u2 = plt.subplots()
    
    figs = [fig_mse_u1, fig_mse_u2, fig_kt_u1, fig_kt_u2]
    axes = [ax_mse_u1, ax_mse_u2, ax_kt_u1, ax_kt_u2]

    for e in exper_types:
        # --- Process Data for User 1 ---
        error_val1 = np.array(out_data[e[0]]['user1']['error_vec'])
        error_median1 = np.median(error_val1, axis=0)
        error_q1_1 = np.quantile(error_val1, 0.25, axis=0)
        error_q3_1 = np.quantile(error_val1, 0.75, axis=0)
        
        kt_val1 = np.array(out_data[e[0]]['user1']['kt_vec'])
        kt_median1 = np.median(kt_val1, axis=0)
        kt_q1_1 = np.quantile(kt_val1, 0.25, axis=0)
        kt_q3_1 = np.quantile(kt_val1, 0.75, axis=0)
        
        # --- Process Data for User 2 ---
        error_val2 = np.array(out_data[e[0]]['user2']['error_vec'])
        error_median2 = np.median(error_val2, axis=0)
        error_q1_2 = np.quantile(error_val2, 0.25, axis=0)
        error_q3_2 = np.quantile(error_val2, 0.75, axis=0)

        kt_val2 = np.array(out_data[e[0]]['user2']['kt_vec'])
        kt_median2 = np.median(kt_val2, axis=0)
        kt_q1_2 = np.quantile(kt_val2, 0.25, axis=0)
        kt_q3_2 = np.quantile(kt_val2, 0.75, axis=0)
        
        # Plot User 1 MSE
        ax_mse_u1.plot(x, error_median1, linewidth=4, label=e[0])
        ax_mse_u1.fill_between(x, error_q1_1, error_q3_1, alpha=0.2)
        
        # Plot User 2 MSE
        ax_mse_u2.plot(x, error_median2, linewidth=4, label=e[0])
        ax_mse_u2.fill_between(x, error_q1_2, error_q3_2, alpha=0.2)

        # Plot User 1 KT
        ax_kt_u1.plot(x, kt_median1, linewidth=4, label=e[0])
        ax_kt_u1.fill_between(x, kt_q1_1, kt_q3_1, alpha=0.2)
        
        # Plot User 2 KT
        ax_kt_u2.plot(x, kt_median2, linewidth=4, label=e[0])
        ax_kt_u2.fill_between(x, kt_q1_2, kt_q3_2, alpha=0.2)

    # --- Configure and Save Plots ---
    plot_configs = [
        {'ax': ax_mse_u1, 'fig': fig_mse_u1, 'ylabel': 'MSE', 'title': 'User 1 MSE', 'file': 'mse_plot_user1.png', 'yscale': 'log'},
        {'ax': ax_mse_u2, 'fig': fig_mse_u2, 'ylabel': 'MSE', 'title': 'User 2 MSE', 'file': 'mse_plot_user2.png', 'yscale': 'log'},
        {'ax': ax_kt_u1, 'fig': fig_kt_u1, 'ylabel': 'KT dist', 'title': 'User 1 KT Distance', 'file': 'kt_plot_user1.png'},
        {'ax': ax_kt_u2, 'fig': fig_kt_u2, 'ylabel': 'KT dist', 'title': 'User 2 KT Distance', 'file': 'kt_plot_user2.png'}
    ]

    for config in plot_configs:
        ax = config['ax']
        fig = config['fig']
        ax.set_xlabel('Number of queries')
        ax.set_ylabel(config['ylabel'])
        if 'yscale' in config:
            ax.set_yscale(config['yscale'])
        ax.set_title(f"{config['title']} | D: {D}, N: {N}, k: {k}")
        leg = ax.legend(prop={'size': 20})
        leg.get_frame().set_facecolor('none')
        leg.get_frame().set_linewidth(0.0)
        
        fig.set_size_inches(12, 9)
        fig.savefig(filePath + '/' + config['file'], dpi=100, bbox_inches='tight')
        

def run_experiment(adaptive, W_sim_pair, seed):
    W_sim1, W_sim2 = W_sim_pair[0, :], W_sim_pair[1, :]

    # A single estimator now manages both users
    estimator = ActiveEstimator()
    estimator.initialize(Embedding, k, adaptive, seed, modelPath, filePath, bounds, pair_samp_rate)
   
    def oracle(p, W_sim_user):
        (a, tau, b) = pair2hyperplane(p)
        num = k*(np.dot(a, W_sim_user) - tau)
        x = num
        y = int(np.random.binomial(1, sp.special.expit(x)))
        return {'y': y, 'x': x, 'a': a, 'tau': tau, 'b': b}

    W_hist1 = np.zeros((M+1, D))
    W_hist2 = np.zeros((M+1, D))
    timer_vec = np.zeros(M+1)

    tic = time.time()
    num_errors1 = 0
    num_errors2 = 0

    for i in range(0, M+1):
        # Get estimates for both users from the single estimator
        W_est1, W_est2 = estimator.getEstimates()
        W_hist1[i, :] = W_est1
        W_hist2[i, :] = W_est2
        timer_vec[i] = time.time() - tic

        if i == M:
            break

        print('measurement {} / {}'.format(i+1, M))

        # Update posteriors for both users (handled internally by the method)
        estimator.update_posterior()
        
        # Select a single query based on both posteriors
        query = estimator.select_query()
        
        # Get oracle responses for both users
        oracle_out1 = oracle(query, W_sim1)
        oracle_out2 = oracle(query, W_sim2)

        # Update the estimator with both observations
        estimator.add_observation(query, oracle_out1, oracle_out2)

        if oracle_out1['y'] != (oracle_out1['x'] > 0):
            num_errors1 += 1
        if oracle_out2['y'] != (oracle_out2['x'] > 0):
            num_errors2 += 1

    print('User 1: {} / {} individual errors'.format(num_errors1, M))
    print('User 2: {} / {} individual errors'.format(num_errors2, M))

    return (W_hist1, W_hist2), timer_vec

if __name__ == "__main__":
    main()
