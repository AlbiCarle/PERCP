import os
import pickle

# Function to get attack names
def get_attacks_name(attack_configs):
    attack_types = {attack["attack_type"] for attack in attack_configs}
    return "_".join(sorted(attack_types)) if attack_types else ""

# Function to get steps names
def get_steps_name(attack_configs):
    steps_types = [str(attack["steps"]) for attack in attack_configs if "steps" in attack]
    return "_".join(sorted(steps_types)) if steps_types else ""

# Function to get norm (distance metric) names
def get_norm_name(attack_configs):
    norm_types = {attack["distance"] for attack in attack_configs if "distance" in attack}
    return "_".join(sorted(norm_types)) if norm_types else ""

# Function to get epsilon values
def get_eps_name(attack_configs):
    eps_types = {str(attack["epsilon"]) for attack in attack_configs if "epsilon" in attack}
    return "_".join(sorted(eps_types)) if eps_types else ""


# Function to save attack results
def save_attack_results(data, folder, filename):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    
    # Remove existing file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
