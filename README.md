# DALI
# Simultaneous Active Learning of User Preferences

This project simulates an active learning environment to simultaneously estimate the preferences of two users. It uses a Bayesian approach, modeling user preferences as vectors in a high-dimensional space. The system actively selects pairwise comparison queries to efficiently learn these preference vectors and evaluates the performance of different query selection strategies.

-----

## Overview

The core of this project is the `ActiveEstimator` class, which manages the preference estimation for two users. The simulation operates in a loop where, at each step, the estimator refines its belief about each user's preferences, selects an informative query, and updates its beliefs based on simulated user responses.

The primary goal is to compare an active learning strategy (**Alternating Information Gain**) against a baseline (**Random Selection**) to see how efficiently each can learn the users' true preferences.

### Key Features

  * **Simultaneous Two-User Estimation**: A single `ActiveEstimator` instance manages the models, data, and posterior distributions for two distinct users.
  * **Bayesian Inference**: User preferences are estimated using a logistic preference model. The posterior distribution is sampled using MCMC (via PyStan), providing a robust estimate of the preference vectors.
  * **Active Query Selection**: Implements multiple strategies for choosing the next item pair to present to the users:
      * **Random**: A baseline strategy that selects a random pair of items.
      * **Information Gain **: An active strategy that, for each query, maximizes the sum of the information gain for User 1 and User 2.
      * **Uncertainty **: An active strategy that, for each query, maximizes the sum of the uncertainty for User 1 and User 2.
      * **Alternating Information Gain **: An active strategy that, for each query, alternates between maximizing the  information gain for User 1 and User 2.
      * **Alternating Uncertainty **: An active strategy that, for each query, alternates between maximizing the uncertainty for User 1 and User 2.
      
  * **Performance Evaluation**: The accuracy of the preference estimates is tracked throughout the simulation using two key metrics:
      * **Mean Squared Error (MSE)**: The squared Euclidean distance between the estimated and true preference vectors.
      * **Normalized Kendall Tau (KT) Distance**: A rank-based metric that measures the disagreement in the preference ordering of items between the estimate and the ground truth.


-----

## File Structure

  * `run_experiments_two_users.py`: This is the main executable script. It sets up the experiment parameters (dimensionality, number of items, etc.), runs the simulation loop, and handles the saving and plotting of results.
  * `active_estimate_two_users.py`: This file contains the core logic. The `ActiveEstimator` class defines the Bayesian model, manages the MCMC sampling, and implements the query selection algorithms.

-----

## Dependencies

To run this project, you will need Python 3.9 and the following libraries. The most critical dependency is `pystan`, which may require a C++ compiler to be installed on your system.

  * `numpy`
  * `scipy`
  * `matplotlib`
  * `pystan`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

-----

## How to Run the Experiment

1.  **Install Dependencies**: Make sure all the required libraries are installed by running the command above.

2.  **Configure Parameters (Optional)**: You can adjust the simulation parameters at the top of the `run_experiments_logistic_model_diff_methods.py` file.

    ```python
    D = 100       # Dimension of the preference space
    N = 1000      # Total number of items
    M = 100       # Number of queries to perform
    ntrials = 1   # Number of independent trials to run
    methods = ['Random', 'Info-NN'] # Methods to compare
    ```

3.  **Execute the Script**: Run the main script from your terminal.

    ```bash
    python run_experiments_logistic_model_diff_methods.py
    ```

### How It Works

The simulation follows these steps for each trial:

1.  **Initialization**: Two "ground truth" preference vectors (`W_sim`) are randomly generated. A single `ActiveEstimator` is initialized to learn them.
2.  **Query Loop**: For `M` iterations, the estimator performs the following actions:
      * **Update Posterior**: It runs MCMC sampling for each user based on all observations collected so far to update its belief (posterior distribution) about their preference vector.
      * **Select Query**: It selects a single pair of items to ask about. If using the `Info-NN` method, it alternates between choosing the pair that is most informative for User 1 and the pair most informative for User 2.
      * **Simulate Oracle**: The query is shown to two simulated "oracles," each of which provides a response based on their ground truth preference vector and a logistic noise model.
      * **Add Observation**: The responses from both users are passed back to the `ActiveEstimator`, which stores this new information for the next posterior update.
3.  **Logging & Evaluation**: At each step, the current estimates for both users' preference vectors are recorded and compared against the ground truth using MSE and KT Distance.

-----

## Output

After running, the script will create a new timestamped directory inside the `./Result/` folder. For example: `./Result/2025_09_17_04_24_12/`. This directory will contain:

  * **`run_info`**: A text file logging the experiment parameters and the console output from the Stan MCMC sampler.
  * **`results.npy`**: A NumPy data file containing the raw results, including the history of preference estimates, errors, and KT distances for each trial and method.
  * **Plot Images**: Four `.png` files containing the performance plots:
      * `mse_plot_user1.png`: MSE vs. queries for User 1.
      * `mse_plot_user2.png`: MSE vs. queries for User 2.
      * `kt_plot_user1.png`: KT Distance vs. queries for User 1.
      * `kt_plot_user2.png`: KT Distance vs. queries for User 2.

