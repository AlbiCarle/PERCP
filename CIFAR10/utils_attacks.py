import secml
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from joblib import Parallel, delayed
import pickle
import os
from secml.array import CArray


# PGD (-1,l2,linf)

from secml.adv.attacks.evasion import CFoolboxPGD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
def PGD_attack(x, y, clf, y_target=None, epsilon=0.2, distance = 'l2', step_size=0.01, steps=100, lb=0.0, ub=1.0):
        
    pgd_attack = CFoolboxPGD(clf,
                               y_target = y_target,
                               lb=lb,
                               ub=ub,
                               epsilons=epsilon,
                               distance = distance,
                               abs_stepsize=step_size,
                               steps=steps,
                               random_start=False)
  
    _, _, x_tilde, _ = pgd_attack.run(x, y)
    return x_tilde.X

"""


# +
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def PGD_attack(x, y, clf, y_target=None, epsilon=0.2, distance='l2', step_size=0.01, steps=100, lb=0.0, ub=1.0):
    
    if epsilon == 0:
        
        x_adv = x
        
    else:
    
        pgd_attack = CFoolboxPGD(clf,
                                 y_target=y_target,
                                 lb=lb,
                                 ub=ub,
                                 epsilons=epsilon,
                                 distance=distance,
                                 abs_stepsize=step_size,
                                 steps=steps,
                                 random_start=False)

        _, _, x_adv, _ = pgd_attack.run(x, y)
                        
        perturbation = x_adv.X - x

        if distance == 'l2':
            norm = np.linalg.norm(perturbation.tondarray(), ord=2)
            if norm > 0:
                perturbation = perturbation * (epsilon / norm)  # Rescale for L2
        elif distance == 'linf':
            perturbation = np.clip(perturbation.tondarray(), -epsilon, epsilon)  # Clip for L_inf
        elif distance == 'l1':
            norm = np.linalg.norm(perturbation.tondarray(), ord=1)
            if norm > 0:
                perturbation = perturbation * (epsilon / norm)  # Rescale for L1

        x_adv = x + CArray(perturbation)
        x_adv = CArray(np.clip(x_adv.tondarray(), lb, ub))

    return x_adv



# -

def run_PGD(db, desc, clf, epsilon, distance, step_size, steps, lb, ub):
    """
    Run PGD attack on the entire dataset in parallel using joblib.
    Ensures the result is always a list.
    """
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y  
        return PGD_attack(x0, y0, clf, None, epsilon, distance, step_size, steps, lb, ub)
    
    att_db_PGD = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_PGD)

# FGM (-1,l2,linf)

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_fgm_attack import CFoolboxFGM

def FGM_attack(x, y, clf, y_target=None, epsilon=0.2, distance='l2', lb=0.0, ub=1.0):
  
    fgm_attack = CFoolboxFGM(clf,
                              y_target=y_target,
                              lb=lb,
                              ub=ub,
                              epsilons=epsilon,
                              distance=distance,
                              random_start=False)
    
    _, _, x_tilde, _ = fgm_attack.run(x, y)
    return x_tilde.X

def run_FGM(db, desc, clf, epsilon, distance, lb = 0.0, ub = 1.0):
   
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y  
        return FGM_attack(x0, y0, clf, None, epsilon, distance, lb, ub)
    
    att_db_FGM = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in range(db.X.shape[0])#tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_FGM)

# BasicIterative (-1,l2,linf)

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_basic_iterative_attack import CFoolboxBasicIterative

def BasicIterative_attack(x, y, clf, y_target=None, epsilon=0.2, distance='l2', lb=0.0, ub=1.0, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True):

    bi_attack = CFoolboxBasicIterative(clf,
                                       y_target=y_target,
                                       lb=lb,
                                       ub=ub,
                                       epsilons=epsilon,
                                       distance=distance,
                                       rel_stepsize=rel_stepsize,
                                       abs_stepsize=abs_stepsize,
                                       steps=steps,
                                       random_start=random_start)
    
    _, _, x_tilde, _ = bi_attack.run(x, y)
    return x_tilde.X

def run_BasicIterative(db, desc, clf, epsilon, distance, lb, ub, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True):

    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y  
        return BasicIterative_attack(x0, y0, clf, None, epsilon, distance, lb, ub, rel_stepsize, abs_stepsize, steps, random_start)
    
    att_db_BasicIterative = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_BasicIterative)
# cl_att = run_BasicIterative(cl, 'Attacking calibration set...', clf, epsilon = epsilon, distance = 'linf', lb=0.0, ub=1.0)

# DeepFool (l2,linf)

from secml.adv.attacks.evasion.foolbox.fb_attacks.fb_deepfool_attack import CFoolboxDeepfool

def DeepFool_attack(x, y, clf, y_target=None, epsilon=0.2, distance='l1', lb=0.0, ub=1.0, steps=50, candidates=10, overshoot=0.02, loss='logits'):
    """
    Performs a DeepFool attack using Foolbox.
    """
    df_attack = CFoolboxDeepfool(clf,
                                 y_target=y_target,
                                 lb=lb,
                                 ub=ub,
                                 epsilons=epsilon,
                                 distance=distance,
                                 steps=steps,
                                 candidates=candidates,
                                 overshoot=overshoot,
                                 loss=loss)
    
    _, _, x_tilde, _ = df_attack.run(x, y)
    return x_tilde.X

def run_DeepFool(db, desc, clf, epsilon, distance, lb, ub, steps=50, candidates=10, overshoot=0.02, loss='logits'):
    """
    Run DeepFool attack on the entire dataset in parallel using joblib.
    Ensures the result is always a list.
    """
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y  
        return DeepFool_attack(x0, y0, clf, None, epsilon, distance, lb, ub, steps, candidates, overshoot, loss)
    
    att_db_DeepFool = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in range(db.X.shape[0])#tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_DeepFool)
#cl_att = run_DeepFool(cl, 'Attacking calibration set...', clf, epsilon = epsilon, distance = 'l2', lb=0.0, ub=1.0)

# CW (l2)

from secml.adv.attacks.evasion import CFoolboxL2CarliniWagner

def CW_attack(x, y, clf, y_target=None, step_size=0.05, steps=100, lb=0.0, ub=1.0):
    """
    Function to perform the Carlini-Wagner (CW) attack on an input sample.
    """
    cw_attack = CFoolboxL2CarliniWagner(
        clf,
        y_target=y_target,
        lb=lb,
        ub=ub,
        steps=steps,
        binary_search_steps=9,
        stepsize=step_size,
        abort_early=True
    )
    
    # Run the attack and return the adversarial sample
    _, _, x_tilde, _ = cw_attack.run(x, y)
    return x_tilde.X

def run_CW(db, desc, clf, step_size, steps, lb, ub):
    """
    Run CW attack on the entire dataset in parallel using joblib.
    Ensures the result is always a list.
    """
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y
        return CW_attack(x0, y0, clf, None, step_size, steps, lb, ub)
    
    att_db_CW = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_CW)
#run_CW(cl, 'Attacking calibration set...', clf, step_size=0.01, steps=100, lb=0.0, ub=1.0)

# ### CFoolboxL2DDN¶

from secml.adv.attacks.evasion import CFoolboxL2DDN


def DDN_attack(x, y, clf, y_target=None, epsilon = 0.2, gamma=0.05, init_epsilon = 1.0, steps=10, lb=0.0, ub=1.0):
    """
    Function to perform the Carlini-Wagner (CW) attack on an input sample.
    """
    ddn_attack = CFoolboxL2DDN(
        clf,
        y_target=y_target,
        lb=lb,
        ub=ub,
        epsilons=epsilon,
        init_epsilon=init_epsilon,
        steps=steps,
        gamma = gamma
    )
    
    # Run the attack and return the adversarial sample
    _, _, x_tilde, _ = ddn_attack.run(x, y)
    return x_tilde.X


def run_DDN(db, desc, clf, epsilon, gamma, init_epsilon, steps, lb, ub):
    """
    Run CW attack on the entire dataset in parallel using joblib.
    Ensures the result is always a list.
    """
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y
        return DDN_attack(x0, y0, clf, None, epsilon, gamma, init_epsilon, steps, lb, ub)
    
    att_db_DDN = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_DDN)



# #### CFoolboxEAD

from secml.adv.attacks.evasion import CFoolboxEAD


def EAD_attack(x, y, clf, y_target = None, epsilon = 0.2, binary_search_steps=15, steps=50, initial_stepsize=0.01, confidence=0.0, initial_const=0.01, regularization=0.1, decision_rule = 'EN', abort_early=True,  lb=0.0, ub=1.0):

    ead_attack = CFoolboxEAD(
        clf,
        y_target=y_target,
        lb=lb,
        ub=ub,
        epsilons=epsilon,
        binary_search_steps = binary_search_steps, 
        steps=steps,
        initial_stepsize = initial_stepsize, 
        confidence = confidence,
        initial_const=initial_const, 
        regularization=regularization, 
        decision_rule = 'EN', 
        abort_early=True
        )
    
    _, _, x_tilde, _ = ead_attack.run(x, y)
    return x_tilde.X



def run_EAD(db, desc, clf, epsilon, steps, lb, ub):
    """
    Run CW attack on the entire dataset in parallel using joblib.
    Ensures the result is always a list.
    """
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y
        return DDN_attack(x0, y0, clf, None, epsilon, steps, lb, ub)
    
    att_db_EAD = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )
    
    return list(att_db_DDN)



# Combine all the attacks

# Define attack functions dictionary
attack_functions = {
    "PGD": PGD_attack,
    "FGM": FGM_attack,
    "BasicIterative": BasicIterative_attack,
    "DeepFool":DeepFool_attack,
    "CW": CW_attack,
    "DDN":DDN_attack,
    "EAD":EAD_attack
}

attack_params = {
        "PGD": ["epsilons", "distance", "step_size", "steps", "lb", "ub"],
        "FGM": ["epsilons", "distance", "lb", "ub"],
        "DeepFool": ["epsilons", "distance", "lb", "ub", "steps"],
        "BasicIterative": ["epsilons", "distance", "lb", "ub", "steps"],
        "CW": ["step_size", "steps"],
        "DDN": ["epsilons","gamma", "init_epsilon", "steps", "lb", "ub"],
        "EAD": ["epsilons", "steps", "binary_search_steps", "initial_stepsize", "confidence", "initial_const", "regularization", "lb", "ub"],
    }


def perform_attack(x, y, clf, attack_type, **kwargs):
    
    if attack_type not in attack_functions:
        raise ValueError(f"Unknown attack type: {attack_type}")
        
    valid_kwargs = {k: v for k, v in kwargs.items() if k in attack_params.get(attack_type, [])}

    attack_fn = attack_functions[attack_type]  # Get attack function
    return attack_fn(x, y, clf, **valid_kwargs)  # Run attack

def run_attack(db, desc, clf, attack_type, **kwargs):
    
    def attack_sample(i):
        x0, y0 = db[i, :].X, db[i, :].Y
        return perform_attack(x0, y0, clf, attack_type, **kwargs)

    att_db = Parallel(n_jobs=1)(
        delayed(attack_sample)(i) for i in tqdm(range(db.X.shape[0]), desc=desc, unit="sample")
    )

    return list(att_db)

def get_single_attack_key(attack):
    attack_types = attack["attack_type"] # Use a set to remove duplicates
    # If steps or distance are missing, use None instead of empty string to handle them
    steps_types = attack.get("steps")
    distance_types = attack.get("distance") 
    eps = str(attack.get("epsilon"))
    # Create a composite key: attack name, steps, and norm
    attack_key = f"{attack_types}"

    # Add steps only if present
    if any([steps_types]):  # Only add steps if there is at least one non-None value
        attack_key += "_" + "_".join(map(str, [steps_types]))
    
    # Add distance only if present
    if any([distance_types]):  # Only add distance if there is at least one non-None value
        attack_key += "_" + distance_types
    
    #if any([eps]):  # Only add distance if there is at least one non-None value
    #    if eps != str(None):
    #        attack_key += "_" + eps
            
    return attack_key





# +
"""def find_best_attack(x, y, clf, attack_configs, verbose = False):
    Runs multiple attacks with different parameters and selects the one 
    minimizing classifier confidence in the true class.

    Parameters:
    - x, y: Input sample and its label.
    - clf: Classifier model.
    - attack_configs: List of dictionaries, each specifying an attack type and its parameters.

    Returns:
    - best_x_tilde: Adversarial example with the lowest confidence in the true class.
    best_x_tilde = None
    min_confidence = float("inf")

    for config in attack_configs:
        attack_type = config["attack_type"]
        attack_params = {k: v for k, v in config.items() if k != "attack_type"}

        if attack_type not in attack_functions:
            print(f"Skipping unknown attack: {attack_type}")
            continue  
        #try:
        # Run attack with its specific parameters
        x_tilde = attack_functions[attack_type](x, y, clf, **attack_params)

        # Compute classifier output and confidence
        #logits = clf.decision_function(x_tilde).tondarray()[0]
        #probabilities = softmax((logits - np.mean(logits)) / np.std(logits))
        
        probabilities = clf.decision_function(x_tilde).tondarray()[0]
        
        true_class = int(y.tondarray()[0]) 
        f_x_y = probabilities[true_class]
        
        if verbose:
            print(f"{attack_type} with {attack_params}: Confidence = {f_x_y:.4f}")

        # Track attack with the lowest confidence
        if f_x_y < min_confidence:
            if verbose:
                print(f"New best attack: {attack_type} with {attack_params}")
            min_confidence = f_x_y
            best_x_tilde = x_tilde

        #except Exception as e:
        #    print(f"Error running {attack_type} with {attack_params}: {e}")

    return best_x_tilde
    """

def find_best_attack(x, y, clf, attack_configs, verbose=False):
    """
    Runs multiple attacks with different parameters and selects the one 
    minimizing classifier confidence in the true class, while also storing 
    adversarial examples for each attack.

    Parameters:
    - x, y: Input sample and its label.
    - clf: Classifier model.
    - attack_configs: List of dictionaries, each specifying an attack type and its parameters.

    Returns:
    - attack_samples: Dictionary storing all adversarial examples.
      {"combined": [worst attack sample], attack_1: [...], attack_2: [...], ...}
    """
    
    best_x_tilde = None
    min_confidence = float("inf")
    
    if len(attack_configs) > 1:
        attack_samples = {"combined": None}  # Dictionary to store adversarial samples
    else:
        attack_samples = {}
        
    for i, config in enumerate(attack_configs):
        attack_type = config["attack_type"]
        attack_params = {k: v for k, v in config.items() if k != "attack_type"}

        attack_key = get_single_attack_key(config)

        if attack_type not in attack_functions:
            print(f"Skipping unknown attack: {attack_type}")
            continue

        try:
            # Run attack with its specific parameters
            x_tilde = attack_functions[attack_type](x, y, clf, **attack_params)
            
            # Store the adversarial sample for this attack
            attack_samples[attack_key] = x_tilde

            # Compute classifier output and confidence
            probabilities = clf.decision_function(x_tilde).tondarray()[0]  
            true_class = int(y.tondarray()[0])  
            f_x_y = probabilities[true_class]

            if verbose:
                print(f"{attack_type} with {attack_params}: Confidence = {f_x_y:.4f}")

            # Track attack with the lowest confidence
            if f_x_y < min_confidence:
                if verbose:
                    print(f"New best attack: {attack_type} with {attack_params}")
                min_confidence = f_x_y
                best_x_tilde = x_tilde
                
        except Exception as e:
            print(f"Error running {attack_type} with {attack_params}: {e}")

    if len(attack_configs) > 1:
        # Store the best attack result in the "combined" key
        attack_samples["combined"] = best_x_tilde

    return attack_samples



# -



"""def attack_dataset(dataset, clf, attack_configs, desc="Running attacks", n_jobs=1, verbose = False):
    
    Apply adversarial attacks to an entire dataset in parallel.
    
    Parameters:
    - dataset: A dataset object with dataset.X (samples) and dataset.Y (labels)
    - clf: The classifier model
    - attack_configs: List of attack configurations
    - desc: Description for tqdm progress bar
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    
    Returns:
    - List of adversarial examples, one per dataset sample.
    
    def attack_sample(i):
        x, y = dataset.X[i, :], dataset.Y[i]
        return find_best_attack(x, y, clf, attack_configs, verbose)

    adversarial_examples = Parallel(n_jobs=n_jobs)(
        delayed(attack_sample)(i) for i in tqdm(range(dataset.X.shape[0]), desc=desc, unit="sample", leave=True)
    )

    return list(adversarial_examples)
    """


def attack_dataset(dataset, clf, attack_configs, desc="Running attacks", n_jobs=1, verbose=False):
    """
    Apply adversarial attacks to an entire dataset in parallel.
    
    Parameters:
    - dataset: A dataset object with dataset.X (samples) and dataset.Y (labels)
    - clf: The classifier model
    - attack_configs: List of attack configurations
    - desc: Description for tqdm progress bar
    - n_jobs: Number of parallel jobs (-1 for all CPUs)
    
    Returns:
    - List of dictionaries containing adversarial examples for each sample.
    """
    def attack_sample(i):
        x, y = dataset.X[i, :], dataset.Y[i]
        return find_best_attack(x, y, clf, attack_configs, verbose)

    adversarial_results = Parallel(n_jobs=n_jobs)(
        delayed(attack_sample)(i) for i in tqdm(range(dataset.X.shape[0]), desc=desc, unit="sample", leave=True)
    )
    
    # Convert list of dictionaries into a dictionary of lists
    attack_results = {}
    
    # Initialize keys in the dictionary
    for key in adversarial_results[0].keys():
        attack_results[key] = []
    
    # Populate the lists
    for result in adversarial_results:
        for key, value in result.items():
            attack_results[key].append(value)
    
    if len(attack_configs) > 1:
        return attack_results
    else:
        return attack_results[list(attack_results.keys())[0]]


def get_single_attack_key(attack):
    attack_types = attack["attack_type"] # Use a set to remove duplicates
    # If steps or distance are missing, use None instead of empty string to handle them
    steps_types = attack.get("steps")
    distance_types = attack.get("distance") 
    eps = str(attack.get("epsilon"))
    # Create a composite key: attack name, steps, and norm
    attack_key = f"{attack_types}"

    # Add steps only if present
    if any([steps_types]):  # Only add steps if there is at least one non-None value
        attack_key += "_" + "_".join(map(str, [steps_types]))
    
    # Add distance only if present
    if any([distance_types]):  # Only add distance if there is at least one non-None value
        attack_key += "_" + distance_types
    
    #if any([eps]):  # Only add distance if there is at least one non-None value
    #    if eps != str(None):
    #        attack_key += "_" + eps
            
    return attack_key












