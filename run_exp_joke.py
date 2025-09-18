#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiments with real joke embeddings (D=64, N=100) and Jester ratings (7200×100).
- Reuses ActiveEstimator / AdaptType / pair2hyperplane / normalized_kendall_tau_distance_sampled
- Uses ratings-based oracle (from Jester) instead of synthetic logistic oracle
- Keeps k, pair_samp_rate, candidate selection logic, etc.
- Adds multi-trial accumulation and plotting just like the original run_experiments script
"""

import os
import sys
import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from active_estimate_closed_form_posterior_single_response import (
    ActiveEstimator, AdaptType, pair2hyperplane, normalized_kendall_tau_distance_sampled
)

# ----------------------------
# Data loading helpers
# ----------------------------

def load_embedding(embedding_path):
    """
    Load embeddings from:
      - .npz with keys ('emb', 'ids') OR generic npz (uses first 2D array)
      - .csv with columns: JokeID, embed_0..embed_*
    Returns:
      emb : (N, D) float64 (mean-centered)
      ids : (N,)   object/str/int
    """
    ext = os.path.splitext(embedding_path)[1].lower()
    if ext == ".npz":
        data = np.load(embedding_path, allow_pickle=True)
        if "emb" in data.files:
            emb = np.asarray(data["emb"], dtype=np.float64)
            ids = np.asarray(data["ids"]) if "ids" in data.files else np.arange(emb.shape[0])
        else:
            key = data.files[0]
            arr = np.asarray(data[key])
            if arr.ndim != 2:
                raise ValueError(f"Unexpected npz content in {embedding_path}: shape={arr.shape}")
            emb = arr.astype(np.float64)
            ids = np.arange(emb.shape[0])
    elif ext == ".csv":
        df = pd.read_csv(embedding_path)
        id_col = "JokeID" if "JokeID" in df.columns else df.columns[0]
        embed_cols = [c for c in df.columns if str(c).startswith("embed_")]
        if not embed_cols:
            raise ValueError("CSV embedding must contain columns named embed_0, embed_1, ...")
        emb = df[embed_cols].to_numpy(dtype=np.float64)
        ids = df[id_col].to_numpy()
    else:
        raise ValueError(f"Unsupported embedding format: {embedding_path}")

    emb = emb - emb.mean(axis=0, keepdims=True)  # match your original preprocessing
    return emb, ids


def load_jester_ratings(ratings_path):
    """
    Loads Jester dense ratings: U rows × 100 columns (no header).
    Returns:
      R : (U, N) float64
    """
    df = pd.read_csv(ratings_path, header=None)
    R = df.to_numpy(dtype=np.float64)
    return R


# ----------------------------
# Utility + Oracle
# ----------------------------

def map_pair_to_indices(pair_embed, Embedding):
    """
    Map a (2, D) pair of item embeddings to their row indices in Embedding via nearest L2.
    Robust to float copying.
    """
    v0, v1 = pair_embed[0], pair_embed[1]
    i = int(np.argmin(((Embedding - v0) ** 2).sum(axis=1)))
    j = int(np.argmin(((Embedding - v1) ** 2).sum(axis=1)))
    return i, j


def make_ratings_oracle(R, Embedding, users_pair, rng):
    """
    ratings-based oracle compatible with estimator.add_observation():
      oracle(p, _, user_idx) -> {'y': 0 or 1}
    It compares the two jokes' Jester ratings for the specified user.
    """
    def oracle(p, _unused_W_sim_user, user_idx=None):
        if user_idx is None:
            raise RuntimeError("ratings_oracle requires user_idx argument.")
        i, j = map_pair_to_indices(p, Embedding)
        u_global = users_pair[user_idx]
        diff = R[u_global, i] - R[u_global, j]
        if diff == 0:
            y = int(rng.random() < 0.5)
        else:
            y = int(diff > 0)  # prefer item 0 if it has higher rating
        return {"y": y}
    return oracle


def ridge_ls_W(Embedding, r_user, ridge=1e-6):
    """
    LS estimate W_hat s.t. Embedding @ W ≈ ratings_user
    Used as 'ground truth' proxy for diagnostics (MSE, Kendall-tau).
    """
    D = Embedding.shape[1]
    A = Embedding.T @ Embedding + ridge * np.eye(D)
    b = Embedding.T @ r_user
    return np.linalg.solve(A, b)


# ----------------------------
# Single-run experiment (ratings-based oracle)
# ----------------------------

def run_experiment_with_ratings(adaptive, W_sim_pair, mu_sim_pair, cov_sim_pair,
                                seed, Embedding, R, users_pair,
                                M, k, pair_samp_rate, modelPath, filePath, rng):
    """
    Mirrors your original run_experiment loop, but uses ratings-based oracle.
    Also PRIMES the estimator posteriors so G_samples exists before first select_query().
    Returns:
      (W_hist1, W_hist2), timer_vec
    """
    estimator = ActiveEstimator()
    estimator.initialize(
        Embedding, k, adaptive, seed, modelPath, filePath,
        mu_sim_pair, cov_sim_pair, pair_samp_rate
    )

    # PRIME from prior so W_samples and G_samples are available
    estimator.update_posterior(0)
    estimator.update_posterior(1)

    ratings_oracle = make_ratings_oracle(R, Embedding, users_pair, rng)

    D = Embedding.shape[1]
    W_hist1 = np.zeros((M + 1, D))
    W_hist2 = np.zeros((M + 1, D))
    timer_vec = np.zeros(M + 1)

    tic = time.time()

    # Store initial estimates before any measurement (index 0)
    W_est1, W_est2 = estimator.getEstimates()
    W_hist1[0, :] = W_est1
    W_hist2[0, :] = W_est2
    timer_vec[0] = time.time() - tic

    for i in range(1, M + 1):
        print(f"Measurement {i} / {M}")
        query = estimator.select_query()

        # Randomly choose which user to query (keeps original behavior)
        user_to_query = rng.integers(0, 2)
        print(f"--> Querying User {user_to_query + 1}")

        oracle_out = ratings_oracle(query, None, user_idx=user_to_query)
        estimator.add_observation(query, oracle_out, user_to_query)
        estimator.update_posterior(user_to_query)

        # Log current estimates after this measurement
        W_est1, W_est2 = estimator.getEstimates()
        W_hist1[i, :] = W_est1
        W_hist2[i, :] = W_est2
        timer_vec[i] = time.time() - tic

    return (W_hist1, W_hist2), timer_vec


# ----------------------------
# Main with multi-trial loop + plotting
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", type=str, default="emb_qwen_Qwen-Qwen3-Embedding-0.6B_64d.npz",
                    help="Path to joke embedding file (.npz with keys 'emb','ids' or .csv wide format).")
    ap.add_argument("--ratings", type=str, default="jester-data-1.csv",
                    help="Path to Jester ratings CSV (U×100, no header).")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["MGuided-InfoGain"],
                    choices=["InfoGain","Uncertainty","Alternating-InfoGain","Alternating-Uncertainty","Random","MGuided-InfoGain"],
                    help="Which methods to run (reuse your original labels).")
    ap.add_argument("--M", type=int, default=100, help="Number of queries per trial.")
    ap.add_argument("--pair_samp_rate", type=float, default=0.1, help="Candidate pair sub-sampling rate.")
    ap.add_argument("--k", type=float, default=5.0, help="Noise constant in logistic model (kept for posterior).")
    ap.add_argument("--seed", type=int, default=2020, help="Base random seed.")
    ap.add_argument("--ntrials", type=int, default=5, help="Number of trials for aggregation/plots.")
    ap.add_argument("--outdir", type=str, default="./Result", help="Output base dir.")
    args = ap.parse_args()

    # Load data
    Embedding, ids = load_embedding(args.embedding)
    R = load_jester_ratings(args.ratings)

    N, D = Embedding.shape
    assert N == 100, f"Expected N=100 jokes; got N={N}"
    assert R.shape[1] == N, f"Ratings columns ({R.shape[1]}) must match N={N}"

    # Methods list (reuse your enums)
    exper_types = [

        ["InfoGain", AdaptType.INFOGAIN],
        ["Uncertainty", AdaptType.UNCERTAINTY],
        ["Alternating-InfoGain", AdaptType.ALTERNATING_INFOGAIN],
        ["Alternating-Uncertainty", AdaptType.ALTERNATING_UNCERTAINTY],
        ["Random", AdaptType.RANDOM],
        ["MGuided-InfoGain", AdaptType.INFOGAIN_MIDPOINT],
    ]
    exper_types = [e for e in exper_types if e[0] in args.methods]

    # Output folder (timestamped)
    timeStamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    modelPath = args.outdir
    filePath = os.path.join(args.outdir, timeStamp)
    os.makedirs(filePath, exist_ok=True)

    # Seeds per trial
    seeds = [args.seed + t for t in range(args.ntrials)]

    M = args.M
    k = args.k
    pair_samp_rate = args.pair_samp_rate

    # ---------------------------------
    # Aggregator like your original code
    # ---------------------------------
    out_data = {}
    for e in exper_types:
        out_data[e[0]] = {
            'user1': {'W_hist': [], 'W_sim': [], 'timer_vec': [], 'error_vec': [], 'kt_vec': []},
            'user2': {'W_hist': [], 'W_sim': [], 'timer_vec': [], 'error_vec': [], 'kt_vec': []}
        }

    # Trials
    for trial, seed in enumerate(seeds, 1):
        print(f"\n=== Trial {trial}/{args.ntrials} | seed={seed} ===")
        rng = np.random.default_rng(seed)

        # Randomly select a pair of users for this trial
        num_users = R.shape[0]
        users_pair = rng.choice(num_users, size=2, replace=False)
        print(f"Selected users: global indices {users_pair.tolist()}")

        # LS “ground-truth” proxies for diagnostics
        W_sim1 = ridge_ls_W(Embedding, R[users_pair[0], :])
        W_sim2 = ridge_ls_W(Embedding, R[users_pair[1], :])
        W_sim_pair = np.stack([W_sim1, W_sim2], axis=0)

        # Keep prior mean/cov simple (zeros/I)
        mu_sim_pair = np.stack([np.zeros(D), np.zeros(D)], axis=0)
        cov_sim_pair = np.stack([np.eye(D), np.eye(D)], axis=0)

        for e in exper_types:
            print('experiment', e[0])
            (W_hist1, W_hist2), timer_vec = run_experiment_with_ratings(
                e[1], W_sim_pair, mu_sim_pair, cov_sim_pair, seed,
                Embedding, R, users_pair,
                M, k, pair_samp_rate, modelPath, filePath, rng
            )

            # Process results for User 1
            error1 = (np.linalg.norm(W_hist1 - W_sim1, axis=1) ** 2) / (np.linalg.norm(W_sim1) ** 2 + 1e-12)
            kt_dist1 = normalized_kendall_tau_distance_sampled(W_hist1, W_sim1, Embedding)
            o1 = out_data[e[0]]['user1']
            o1['W_hist'].append(W_hist1); o1['W_sim'].append(W_sim1); o1['timer_vec'].append(timer_vec)
            o1['error_vec'].append(error1); o1['kt_vec'].append(kt_dist1)

            # Process results for User 2
            error2 = (np.linalg.norm(W_hist2 - W_sim2, axis=1) ** 2) / (np.linalg.norm(W_sim2) ** 2 + 1e-12)
            kt_dist2 = normalized_kendall_tau_distance_sampled(W_hist2, W_sim2, Embedding)
            o2 = out_data[e[0]]['user2']
            o2['W_hist'].append(W_hist2); o2['W_sim'].append(W_sim2); o2['timer_vec'].append(timer_vec)
            o2['error_vec'].append(error2); o2['kt_vec'].append(kt_dist2)
            print()

    # Save numpy dump like your original
    np.save(os.path.join(filePath, 'results_random_user.npy'), out_data)

    # -----------------
    # Plotting (4 figs)
    # -----------------
    x = range(0, M + 1)

    fig_mse_u1, ax_mse_u1 = plt.subplots()
    fig_mse_u2, ax_mse_u2 = plt.subplots()
    fig_kt_u1,  ax_kt_u1  = plt.subplots()
    fig_kt_u2,  ax_kt_u2  = plt.subplots()

    for e in exper_types:
        # User 1
        error_val1 = np.array(out_data[e[0]]['user1']['error_vec'])  # [ntrials, M+1]
        ax_mse_u1.plot(x, np.median(error_val1, axis=0), linewidth=4, label=e[0])
        ax_mse_u1.fill_between(x, np.quantile(error_val1, 0.25, axis=0), np.quantile(error_val1, 0.75, axis=0), alpha=0.2)

        kt_val1 = np.array(out_data[e[0]]['user1']['kt_vec'])
        ax_kt_u1.plot(x, np.median(kt_val1, axis=0), linewidth=4, label=e[0])
        ax_kt_u1.fill_between(x, np.quantile(kt_val1, 0.25, axis=0), np.quantile(kt_val1, 0.75, axis=0), alpha=0.2)

        # User 2
        error_val2 = np.array(out_data[e[0]]['user2']['error_vec'])
        ax_mse_u2.plot(x, np.median(error_val2, axis=0), linewidth=4, label=e[0])
        ax_mse_u2.fill_between(x, np.quantile(error_val2, 0.25, axis=0), np.quantile(error_val2, 0.75, axis=0), alpha=0.2)

        kt_val2 = np.array(out_data[e[0]]['user2']['kt_vec'])
        ax_kt_u2.plot(x, np.median(kt_val2, axis=0), linewidth=4, label=e[0])
        ax_kt_u2.fill_between(x, np.quantile(kt_val2, 0.25, axis=0), np.quantile(kt_val2, 0.75, axis=0), alpha=0.2)

    plot_configs = [
        {'ax': ax_mse_u1, 'fig': fig_mse_u1, 'ylabel': 'MSE',     'title': 'User 1 MSE',        'file': 'mse_plot_user1.png', 'yscale': 'log'},
        {'ax': ax_mse_u2, 'fig': fig_mse_u2, 'ylabel': 'MSE',     'title': 'User 2 MSE',        'file': 'mse_plot_user2.png', 'yscale': 'log'},
        {'ax': ax_kt_u1,  'fig': fig_kt_u1,  'ylabel': 'KT dist', 'title': 'User 1 KT Distance','file': 'kt_plot_user1.png'},
        {'ax': ax_kt_u2,  'fig': fig_kt_u2,  'ylabel': 'KT dist', 'title': 'User 2 KT Distance','file': 'kt_plot_user2.png'}
    ]

    for config in plot_configs:
        ax, fig = config['ax'], config['fig']
        ax.set_xlabel('Number of queries'); ax.set_ylabel(config['ylabel'])
        if 'yscale' in config: ax.set_yscale(config['yscale'])
        ax.set_title(f"{config['title']} | D: {D}, N: {N}, k: {k}")
        leg = ax.legend(prop={'size': 12})
        leg.get_frame().set_facecolor('none'); leg.get_frame().set_linewidth(0.0)
        fig.set_size_inches(12, 9)
        out_png = os.path.join(filePath, config['file'])
        fig.savefig(out_png, dpi=100, bbox_inches='tight')
        print(f"Saved {out_png}")

    # Also save a compact JSON summary
    out_json = os.path.join(filePath, "results_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "embedding_path": args.embedding,
                "ratings_path": args.ratings,
                "N": int(N), "D": int(D),
                "M": int(M),
                "pair_samp_rate": pair_samp_rate,
                "k": k,
                "ntrials": int(args.ntrials),
                "methods": [e[0] for e in exper_types],
                "seeds": seeds,
            }
        }, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
