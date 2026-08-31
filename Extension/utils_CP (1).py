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
        
        logits = clf.decision_function(x).tondarray()[0]
        #logits = clf.decision_function(x).tondarray()[0]
        #logits_normalized = (logits - np.mean(logits)) / np.std(logits)

        #probabilities = softmax(logits)#softmax(logits_normalized)
        probabilities = logits#softmax(logits, dim=0)
        
        return 1 - probabilities[int(y.tondarray()[0])]

# Additional functions

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

# Compute CP sets

# +
digits = [0,1,2,3,4,5,6,7,8,9]
dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def compute_CP(db, qhat, clf, method = 'lac'):
    
    conformal_sets = []  # List to store conformal sets for each test point
    
    if isinstance(db, list):
        n = len(db)
    else:
        n = db.X.shape[0]

    # Wrap the loop with tqdm to show progress
    for i in tqdm(range(n), desc="Computing CP sets", ncols=100):
        # Initialize the conformal set for this sample
        if isinstance(db, list):
            x = db[i]
        else:
            x = db[i,:].X
        
        conformal_set = []
        
        # Check each class digit
        for d in digits:
            # Compute the score function for the current digit
            score = s(x, CArray([d]), clf, method)
            
            # Add the digit to the conformal set if the score satisfies the threshold
            if score <= qhat:
                conformal_set.append(d)
        
        # Store the conformal set for this sample
        conformal_sets.append(conformal_set)

    return conformal_sets


# -

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

def compute_set_sizes(dataset, conformal_sets, verbose = True):
    
    conformal_set_sizes = [len(conformal_set) for conformal_set in conformal_sets]

    max_size = 10  # Maximum size to consider
    counts = {size: conformal_set_sizes.count(size) for size in range(max_size + 1)}

    # Print results
    if verbose:
        for size, count in counts.items():
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

# Mean and variance conformal sets

def mean_conformal_sets(conformal_sets):

    non_empty_sets = [set for set in conformal_sets] #if set

    set_sizes = [len(s) for s in non_empty_sets]
    
    print(f"Average set size: {np.mean(set_sizes)}")

    return np.mean(set_sizes) if set_sizes else 0 

def std_conformal_sets(conformal_sets):
    # Filter out empty sets
    non_empty_sets = [s for s in conformal_sets] # if s
    
    # Compute set sizes
    set_sizes = [len(s) for s in non_empty_sets]
    
    # Calculate and print the standard deviation
    std_size = np.std(set_sizes) if set_sizes else 0
    print(f"Std set size: {std_size}")
    
    return std_size

# Plots

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

def calibration_curve(db, db2, calibration_scores, clf, alpha1, alpha2, steps):
    
    if isinstance(db, list):

        n = len(db)

    else:

        n = db.X.shape[0]

    scores = []
    
    n_cl = len(calibration_scores)
    
    for i in range(n):
        
        if isinstance(db, list):

            x = db[i]

        else:

            x = db.X[i,:]
            
        scores.append([s(x, CArray([d]), clf) for d in digits]) 
    scores = np.array(scores)  
    labels = np.array(db2.Y.tondarray())
        
    alphas = np.linspace(alpha1, alpha2, steps)
    probabilities = []

    for alpha in alphas:
        q_level = np.ceil((n_cl + 1) * (1 - alpha)) / n_cl
        qhat = np.quantile(calibration_scores, q_level, method='higher')
        conformal_sets = scores <= qhat 

        true_label_in_conformal_set = [
            conformal_sets[i, digits.index(y)] for i, y in enumerate(labels)
        ]    
        
        n_correct = sum(true_label_in_conformal_set)
        probability = n_correct / n
        probabilities.append(probability)

    return probabilities, alphas    

def plot_calibration_curve(db, db2, calibration_scores, clf, alpha1 = 0.1, alpha2 = 0.9, steps = 50):
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

def plot_conformal_analysis(db, db2, conformal_set_sizes, calibration_scores, clf, alpha1= 0.1, alpha2 = 0.9, steps = 50):
    """
    Combines the conformal set size histogram and calibration curve into a single figure with two subplots.
    """
    # Compute the calibration curve values
    probabilities, alphas = calibration_curve(db, db2, calibration_scores, clf, alpha1, alpha2, steps)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Histogram of conformal set sizes
    conformal_set_sizes = [0 if size == 0 else size for size in conformal_set_sizes]
    bins = np.arange(0, max(conformal_set_sizes) + 2) - 0.5  # Add 0 bin
    axes[0].hist(conformal_set_sizes, bins=bins, edgecolor='black', rwidth=0.8, color='blue')
    
    # Highlight the bin for size 0 in red
    axes[0].patches[0].set_facecolor('red')
    axes[0].set_title("Conformal Set Sizes", fontsize=14)
    axes[0].set_xlabel("Set Size (Number of Labels in Conformal Set)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_xticks(np.arange(0, max(conformal_set_sizes) + 2))
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

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




















