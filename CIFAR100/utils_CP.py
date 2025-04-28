# Usual imports
import secml
import numpy as np
from tqdm import tqdm
#from scipy.special import softmax
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from joblib import Parallel, delayed
import pickle
import os
import seaborn as sns

# SecML
from secml.ml.features.normalization import CNormalizerMinMax
from secml.ml.peval.metrics import CMetricAccuracy
from secml.array import CArray
from secml.ml.classifiers import CClassifierPyTorch

# RobustBench
import robustbench
from robustbench.utils import load_model
from secml.utils import fm
from secml import settings

# Score function


def s(x, y, clf, method = 'lac'):
    
    if method == 'lac':
    
        probabilities = clf.decision_function(x).tondarray()[0]

        return 1 - probabilities[int(y.tondarray()[0])]
        
    if method == 'aps':
        
        probabilities = clf.decision_function(x).tondarray()[0]
        
        # Rank labels by decreasing probability
        ranked_indices = np.argsort(probabilities)[::-1]
        # Sum probabilities until reaching the true label
        cum_sum = 0
        for k, label in enumerate(ranked_indices):
            cum_sum += probabilities[label]
            if label == int(y.tondarray()[0]):
                return cum_sum  # Return cumulative score at true label

        return 1.0  # Should never reach here

# Additional functions

# +
def check_natural_property(ts, ts_att, clf, method='aps'):
    """
    Checks whether s(x0_tilde, y0, clf, method) >= s(x0, y0, clf, method)
    holds for all samples in the dataset.
    
    Parameters:
    - ts: Original dataset, assumed to be a structured array or DataFrame with attributes X and Y.
    - ts_att: Transformed dataset (adversarial or perturbed version of ts).
    - clf: Classifier with a decision function.
    - s: Function computing the desired score.
    - method: Scoring method (default is 'aps').
    
    Returns:
    - results: Boolean array indicating whether the inequality holds for each sample.
    """
    n_ts = ts.X.shape[0]
    results = np.zeros(n_ts, dtype=bool)
    count_increased = 0

    for idx in range(n_ts):
        x0 = ts[idx, :].X
        x0_tilde = ts_att[idx]
        y0 = ts[idx, :].Y

        s_x0 = s(x0, y0, clf, method=method)
        s_x0_tilde = s(x0_tilde, y0, clf, method=method)

        results[idx] = s_x0_tilde >= s_x0
        
        if s_x0_tilde >= s_x0:
            count_increased += 1

    percentage_increased = (count_increased / n_ts) * 100
    summary = f"The score function has incremented for {count_increased} out of {n_ts} samples ({percentage_increased:.2f}%)."

    print(summary)
    return summary

def compute_score(db, db2, clf, method = 'lac'):
    
    scores = []  

    for i in tqdm(range(db.X.shape[0]), desc="Computing scores...", unit="sample"):
        y = db[i,:].Y

        if isinstance(db2, list):
            
            x = db2[i]
            
        else:
            
            x = db2[i,:].X

        score = s(x,y, clf, method)

        scores.append(score)
    
    return scores  


# -

# Compute CP sets

# +

digits = list(range(100))

def compute_CP(db, qhat, clf, method='lac', verbose=True):
    conformal_sets = []  # List to store conformal sets for each test point
    scores_list = []  # List to store scores for each test point
    
    if isinstance(db, list):
        n = len(db)
    else:
        n = db.X.shape[0]
        
    iterable = range(n) if not verbose else tqdm(range(n), desc="Computing CP sets", ncols=100)

    for i in iterable:
        if isinstance(db, list):
            x = db[i]
        else:
            x = db[i, :].X

        scores = {}  # Dictionary to store scores for all digits
        conformal_set = []

        if method == 'lac':
            for d in digits:
                score = s(x, CArray([d]), clf, method)
                scores[d] = score  # Store score
                if score <= qhat:
                    conformal_set.append(d)

        elif method == 'aps':
            # Get scores for all digits
            scores = {d: s(x, CArray([d]), clf, method) for d in digits}
            # Sort class indices by increasing score (higher confidence first)
            sorted_digits = sorted(scores, key=scores.get)
            # Build APS set: accumulate scores until threshold is reached
            for d in sorted_digits:
                conformal_set.append(d)
                if scores[d] >= qhat:  # Stop when reaching quantile
                    break

        else:
            raise ValueError("Unsupported method. Choose 'lac' or 'aps'.")

        conformal_sets.append(conformal_set)
        scores_list.append(scores)

    return conformal_sets, scores_list

def compute_covergae(dataset, conformal_sets):
    
    true_label_in_conformal_set = []
    n = dataset.X.shape[0]  # Assuming `dataset` is a list or similar iterable
    
    for i in range(n):
        y0 = dataset[i,:].Y
        conformal_set = conformal_sets[i]
        true_label_in_conformal_set.append(y0 in conformal_set)

    n_correct = sum(true_label_in_conformal_set)
    accuracy = n_correct / n * 100
    
    print(f"True label is in the conformal set for {n_correct}/{n} test samples ({accuracy:.2f}%).")
    
    return accuracy


def compute_set_sizes(dataset, conformal_sets, verbose=True):
    conformal_set_sizes = [len(conformal_set) for conformal_set in conformal_sets]

    max_size = 100  # Maximum size to consider
    counts = {size: conformal_set_sizes.count(size) for size in range(max_size + 1)}

    # Print only nonzero counts
    if verbose:
        for size, count in counts.items():
            if count > 0:
                print(f"Number of conformal sets with size {size}: {count}")

    return conformal_set_sizes

def compute_covergae_std(dataset, conformal_sets):
    
    true_label_in_conformal_set = []
    n = dataset.X.shape[0]  # Assuming `dataset` is a list or similar iterable
    
    for i in range(n):
        y0 = dataset[i,:].Y
        conformal_set = conformal_sets[i]
        true_label_in_conformal_set.append(y0 in conformal_set)

    vari = np.var(true_label_in_conformal_set)  # Compute mean coverage
    stdi = np.std(true_label_in_conformal_set)
    
    print(f"Coverage std ({stdi:.2f}).")
    
    return stdi

def mean_conformal_sets(conformal_sets):

    non_empty_sets = [set for set in conformal_sets if set] 

    set_sizes = [len(s) for s in non_empty_sets]
    
    print(f"Average set size: {np.mean(set_sizes)}")

    return np.mean(set_sizes) if set_sizes else 0 

def std_conformal_sets(conformal_sets):
    # Filter out empty sets
    non_empty_sets = [s for s in conformal_sets if s]
    
    # Compute set sizes
    set_sizes = [len(s) for s in non_empty_sets]
    
    # Calculate and print the standard deviation
    std_size = np.std(set_sizes) if set_sizes else 0
    print(f"Std set size: {std_size}")
    
    return std_size

def plot_conformal_histogram(conformal_set_sizes):
    """
    Plots a histogram of conformal set sizes, highlighting the 0 size bin in red.
    """
    # Ensure we include a bin for 0 class size
    conformal_set_sizes = [0 if size == 0 else size for size in conformal_set_sizes]
    bins = np.arange(0, max(conformal_set_sizes) + 2) - 0.5  # Add 0 bin

    plt.figure(figsize=(8, 6))
    plt.hist(conformal_set_sizes, bins=bins, edgecolor='black', rwidth=0.8, color='blue')

    # Highlight the bin for size 0 in red
    ax = plt.gca()
    ax.patches[0].set_facecolor('red')

    plt.title("Conformal Set Sizes", fontsize=14)
    plt.xlabel("Set Size (Number of Labels in Conformal Set)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(np.arange(0, max(conformal_set_sizes) + 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def plot_calibration_curve(db, db2, calibration_scores, clf, alpha1 = 0.1, alpha2 = 0.9, steps = 50, method = 'lac'):
    """
    Plots the calibration curve: true label coverage vs. significance level (alpha).
    """
    probabilities, alphas = calibration_curve(db, db2, calibration_scores, clf, alpha1, alpha2, steps)
    
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, probabilities, marker='o', label='True label probability')
    plt.plot(alphas, 1 - alphas, color='r', linestyle='--', label='Expected coverage (1 - alpha)')
    
    plt.title('True Label Coverage vs. Alpha', fontsize=14)
    plt.xlabel(r'$\alpha$ (Significance level)', fontsize=12)
    plt.ylabel('Probability of true label in conformal set', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def calibration_curve(db, db2, calibration_scores, scores_list, clf, alpha1, alpha2, steps, method):
    """
    Computes the calibration curve for a given model using conformal prediction.
    The method can be 'lac' or 'aps' for LAC or APS methods respectively.
    
    Instead of recomputing CP sets, we filter them dynamically from `scores_list`.
    """
    if isinstance(db, list):
        n = len(db)
    else:
        n = db.X.shape[0]

    labels = np.array(db2.Y.tondarray())

    # Generate alpha values for the calibration curve
    alphas = np.linspace(alpha1, alpha2, steps)
    probabilities = []

    # Compute the calibration curve for different alpha levels
    for alpha in tqdm(alphas, desc="Plotting calibration curve", ncols=100):
        q_level = np.ceil((len(calibration_scores) + 1) * (1 - alpha)) / len(calibration_scores)
        qhat = np.quantile(calibration_scores, q_level, method='lower')

        # Generate conformal sets dynamically based on precomputed scores
        conformal_sets = []
        
        for scores in scores_list:  # Iterate over precomputed scores
            if method == 'lac':
                conformal_set = [d for d, score in scores.items() if score <= qhat]
            elif method == 'aps':
                sorted_digits = sorted(scores, key=scores.get)  # Sort by increasing score
                conformal_set = []
                for d in sorted_digits:
                    conformal_set.append(d)
                    if scores[d] >= qhat:
                        break
            else:
                raise ValueError("Unsupported method. Choose 'lac' or 'aps'.")
            
            conformal_sets.append(conformal_set)

        # Check if the true label is in the conformal set
        true_label_in_conformal_set = [
            1 if labels[i] in conformal_sets[i] else 0
            for i in range(n)
        ]

        # Compute the probability of the true label being in the conformal set
        n_correct = sum(true_label_in_conformal_set)
        probability = n_correct / n
        probabilities.append(probability)

    return probabilities, alphas
    
    
def plot_conformal_analysis(db, db2, conformal_set_sizes, calibration_scores, scores_list, clf, alpha1= 0.1, alpha2 = 0.9, steps = 50, method = 'lac', dataset_name = 'CIFAR10'):
    """
    Combines the conformal set size histogram and calibration curve into a single figure with two subplots.
    """
    
    sns.set_style("whitegrid")
    
    print("Plotting results...")
    # Compute the calibration curve values
    probabilities, alphas = calibration_curve(db, db2, calibration_scores, scores_list, clf, alpha1, alpha2, steps, method)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

     # Subplot 1: Histogram of conformal set sizes
    conformal_set_sizes = [0 if size == 0 else size for size in conformal_set_sizes]
    bins = np.arange(0, max(conformal_set_sizes) + 2) - 0.5  # Ensure correct bin edges

    sns.histplot(conformal_set_sizes, bins=bins, ax=axes[0], color="royalblue", edgecolor="black", alpha=0.8)

    # Highlight the bin for size 0 in red
    bars = axes[0].patches
    if len(bars) > 0:
        bars[0].set_facecolor("red")

    axes[0].set_title(f"Conformal Set Sizes ({dataset_name})", fontsize=14)
    axes[0].set_xlabel("Set Size (Number of Labels in Conformal Set)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_xticks(np.arange(0, max(conformal_set_sizes) + 2, max(1, len(set(conformal_set_sizes)) // 10)))
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    
    # Subplot 2: Calibration curve
    axes[1].plot(alphas, probabilities, marker='o', label='True label probability')
    axes[1].plot(alphas, 1 - alphas, color='r', linestyle='--', label='Expected coverage (1 - alpha)')
    axes[1].set_title('True Label Coverage vs. Alpha', fontsize=14)
    axes[1].set_xlabel(r'$\alpha$ (Significance level)', fontsize=12)
    axes[1].set_ylabel('Probability of true label in conformal set', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()    


