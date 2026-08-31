# +
# ======================================================================
# IMPORTS
# ======================================================================
# -

# Standard
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Vision
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset
import torchvision.datasets as datasets


# Progress bar
from rich.progress import Progress

# RobustBench
from robustbench.utils import load_model

# Attacks
import torchattacks
from autoattack import AutoAttack

# SecML (if you use it later)
from secml.utils import fm
from secml import settings

import yaml

import pandas as pd
import os


# ======================================================================
# UTILS
# ======================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_models_from_config(config):
    device = config["device"]
    model_dir = config["model_dir"]
    norm = config["norm"]
    model_names = config["models"]

    models = {}

    for name in model_names:
        print(f"\nLoading model: {name} ...")
        
        base_model = load_rb_model(
            model_name=name,
            device=device,
            norm=norm,
            model_dir=model_dir
        )

        wrapped = ProbRegressionWrapper(base_model)
        wrapped.to(device)
        wrapped.eval()

        models[name] = wrapped

    print("\nLoaded models:")
    print(list(models.keys()))
    
    return models

def load_imagenet_splits(
    root="/path/to/imagenet",
    n_subset = 10000,
    n_train=2000,
    n_cal=1000,
    n_test=2000,
    seed=0
):
    """
    Carica piccoli split da ImageNet (train/cal/test) in memoria,
    coerente con CIFAR10/CIFAR100.

    - restiutisce tensori X_train, y_train, X_cal, y_cal, X_test, y_test
    - immagini normalizzate in [0,1]
    """
    transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),   # standard for ImageNet models
            T.ToTensor()
        ])

    imagenet_path = "/home/acarlevaro/Sources/albi610/Verifiably_Robust_CP/VRCP_Regression/Extension/IMAGENET/ILSVRC2012_Albi"
    
    full_dataset = datasets.ImageFolder(root=imagenet_path, transform=transform)

    indices = random.sample(range(len(full_dataset)), n_subset)
    subset_dataset = Subset(full_dataset, indices)

    rnd_idx = torch.randperm(len(subset_dataset))
    train_idx = rnd_idx[:n_train]
    cal_idx   = rnd_idx[n_train:n_train + n_cal]
    test_idx  = rnd_idx[n_train + n_cal : n_train + n_cal + n_test]

    # ------------------------
    # Convertiamo tutto in tensori (X,y)
    # ------------------------
    def extract_X_y(dataset, idxs):
        X = []
        y = []
        for i in idxs:
            img, label = dataset[i]
            X.append(img.unsqueeze(0))   # (1,3,224,224)
            y.append(label)
        X = torch.cat(X, dim=0)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

    train_X, train_y = extract_X_y(subset_dataset, train_idx)
    cal_X,   cal_y   = extract_X_y(subset_dataset, cal_idx)
    test_X,  test_y  = extract_X_y(subset_dataset, test_idx)


    return (train_X, train_y), (cal_X, cal_y), (test_X, test_y)

class ProbRegressionWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier  # pretrained classifier

    def forward(self, x):
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs

class ProbToLogitsWrapper(torch.nn.Module):
    def __init__(self, model_prob):
        super().__init__()
        self.model_prob = model_prob  # il tuo modello che restituisce probabilità

    def forward(self, x):
        probs = self.model_prob(x)           # (B, K)
        logits = torch.log(probs + 1e-12)    # conversione sicura
        return logits

def get_correct_probs(model, X, y):
    with torch.no_grad():
        probs = model(X)                # shape: (n, 10)
        correct_probs = probs[range(len(y)), y]  # shape: (n,)
    return correct_probs.to(device)   

output_dir = fm.join(settings.SECML_MODELS_DIR, 'robustbench')

def load_rb_model(model_name, dataset="imagenet", device="cuda" , norm = 'corruptions', model_dir="./models"):
    model = load_model(model_name,dataset=dataset, norm=norm, model_dir=model_dir).to(device)
    return model

def attack_with_autoattack(model_prob, X, y, eps=8/255, norm='L2', bs=32, device='cuda'):
    model_prob.eval()

    # wrapper veloce
    model_for_attack = ProbToLogitsWrapper(model_prob).to(device)
    model_for_attack.eval()

    X = X.to(device)
    y = y.to(device)

    autoattack = AutoAttack(
        model_for_attack,
        norm=norm,
        eps=eps,
        version='custom',          # per avere un singolo attacco
        attacks_to_run=['apgd-ce'], 
        device=device,
        verbose=True
    )

    # riduci il numero di iterazioni
    autoattack.apgd.n_iter = 10
    
    adversarial_examples = autoattack.run_standard_evaluation(X, y, bs=bs)

    return adversarial_examples

def nonconformity_score(y_true, y_pred, u=None):
    """
    Compute nonconformity scores for regression CP, optionally using an uncertainty scalar.

    Args:
        y_true: torch.Tensor of shape (n,)  -- true regression targets
        y_pred: torch.Tensor of shape (n,)  -- predicted regression outputs
        u: torch.Tensor of shape (n,), optional -- per-example uncertainty (e.g., predicted std or residual)
    
    Returns:
        scores: torch.Tensor of shape (n,)  -- nonconformity scores
    """
    residuals = torch.abs(y_pred - y_true)
    
    if u is not None:
        scores = residuals / u
        
    else:
        scores = residuals
    
    return scores

def cp_interval_width(u, qhat):
    """
    Computes conformal prediction interval widths.
    
    Args:
        u:    array-like of shape (n,) -- uncertainty scalar u(x)
        qhat: float                   -- conformal quantile

    Returns:
        width: array of shape (n,) -- interval widths
    """
    u = torch.as_tensor(u, dtype=torch.float32)
    return 2 * u * qhat

def extract_scores(models,
                   cal_X, cal_y,
                   test_X, test_y,
                   datamode="clean"):
    """
    Estrarre score, predizioni e u dal calibration e test set.
    Può lavorare sia su dati clean sia attaccati.

    Args:
        models: dict {name: model}
        cal_X, cal_y: calibration data
        test_X, test_y: test data
        datamode: "clean" oppure "adv"

    Returns:
        results: dict con per-modello:
            probs_cal, y_pred_cal, u_cal, score_cal
            probs_test, y_pred_test, u_test, score_test
    """

    results = {}

    with torch.no_grad():
        for name, model in models.items():
            print(f"\n=== Processing model: {name} ({datamode}) ===")

            # -----------------------------
            # Calibrazione
            # -----------------------------
            probs_cal = model(cal_X)
            y_pred_cal = probs_cal[range(len(cal_y)), cal_y]

            #u_cal = ((y_pred_cal - 1) ** 2)#.mean().item()
            #epsilon = 1e-6
            #u_cal = torch.sqrt((y_pred_cal - 1) ** 2 + epsilon)
            
            residuals = torch.abs(y_pred_cal - 1)  # shape (n_cal,)
            residuals_reshaped = residuals.unsqueeze(0).unsqueeze(0)  # shape (1,1,n_cal)

            # Apply a 1D average pooling with kernel size 5
            smoothed_residuals = F.avg_pool1d(residuals_reshaped, kernel_size=5, stride=1, padding=2)
            u_cal = smoothed_residuals.squeeze()
            
            score_cal = nonconformity_score(
                1.0,
                y_pred_cal.float(),
                u=u_cal
            )

            # -----------------------------
            # Test
            # -----------------------------
            probs_test = model(test_X)
            y_pred_test = probs_test[range(len(test_y)), test_y]

            #u_test = ((y_pred_test - 1) ** 2)#.mean().item()
            #u_test = torch.sqrt((y_pred_test - 1) ** 2 + epsilon)
            residuals = torch.abs(y_pred_test - 1)  # shape (n_cal,)
            residuals_reshaped = residuals.unsqueeze(0).unsqueeze(0)  # shape (1,1,n_cal)

            # Apply a 1D average pooling with kernel size 5
            smoothed_residuals = F.avg_pool1d(residuals_reshaped, kernel_size=5, stride=1, padding=2)
            u_test = smoothed_residuals.squeeze()
            
            score_test = nonconformity_score(
                1.0,
                y_pred_test.float(),
                u=u_test
            )

            # -----------------------------
            # Salvataggio
            # -----------------------------
            results[name] = {
                "probs_cal": probs_cal,
                "y_pred_cal": y_pred_cal,
                "u_cal": u_cal,
                "score_cal": score_cal,

                "probs_test": probs_test,
                "y_pred_test": y_pred_test,
                "u_test": u_test,
                "score_test": score_test,
            }

    return results

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def compute_cp(
    clean_results, 
    adv_results, 
    alpha=0.1, 
    cal_on='clean',     # calibrazione su 'clean' o 'adv'
    eval_on='clean',    # dataset su cui calcolare coverage
    interval_on='test'  # 'test' oppure 'cal'
):
    """
    Calcola CP in modo flessibile.

    clean_results: dizionario score clean
    adv_results:   dizionario score avversari
    cal_on:        set per calibrazione ('clean' o 'adv')
    eval_on:       set per coverage ('clean' o 'adv')
    interval_on:   set su cui costruire gli intervalli ('cal' o 'test')
    """
    
    cp_outputs = {}
    
    for name in clean_results.keys():
        print(f"\n--- CP for model: {name} | Cal: {cal_on} | Eval: {eval_on} | Interval: {interval_on} ---")
        
        clean = clean_results[name]
        adv   = adv_results[name]

        # -------------------------
        # STEP 1 — Calibrazione
        # -------------------------
        source_cal = clean if cal_on=='clean' else adv

        score_cal = _to_numpy(source_cal["score_cal"])
        y_pred_cal = _to_numpy(source_cal["y_pred_cal"]).astype(np.float32)
        u_cal = _to_numpy(source_cal["u_cal"])

        n_cal = len(score_cal)
        q_level = np.floor((n_cal + 1) * (1 - alpha)) / n_cal
        qhat = float(
            torch.quantile(torch.tensor(score_cal, dtype=torch.float32), float(q_level))
        )

        print(f" q̂ (CP): {qhat:.6f}")

        # -------------------------
        # STEP 2 — Scegli set per intervalli
        # -------------------------
        if interval_on == 'test':
            y_pred_interval = clean["y_pred_test"] if eval_on=='clean' else adv["y_pred_test"]
            u_interval      = clean["u_test"]      if cal_on=='clean' else adv["u_test"]
        elif interval_on == 'cal':
            y_pred_interval = clean["y_pred_cal"] if eval_on=='clean' else adv["y_pred_cal"]
            u_interval      = clean["u_cal"]      if cal_on=='clean' else adv["u_cal"]
        else:
            raise ValueError("interval_on must be 'cal' or 'test'")

        y_pred_eval = _to_numpy(y_pred_interval).astype(np.float32)
        u_eval      = _to_numpy(u_interval)

        # -------------------------
        # STEP 3 — Coverage (sempre sul test o cal, deciso da eval_on)
        # -------------------------

        y_true_eval = np.ones_like(y_pred_eval, dtype=np.float32)

        # -------------------------
        # STEP 4 — Costruisci intervalli
        # -------------------------
        radius = qhat * u_eval
        lower  = y_pred_eval - radius
        upper  = y_pred_eval + radius

        coverage = float(np.mean((lower <= y_true_eval) & (y_true_eval <= upper)))
        mean_width = float(np.mean(upper - lower))

        print(f" Coverage: {coverage*100:.2f}%")
        print(f" Mean interval width: {mean_width:.6f}")

        cp_outputs[name] = {
            "qhat": qhat,
            "coverage": coverage,
            "mean_width": mean_width,
            "lower": lower,
            "upper": upper,
            "y_pred_interval": y_pred_interval,
            "u_interval": u_interval,
            "cal_on": cal_on,
            "eval_on": eval_on,
            "interval_on": interval_on,
        }

    return cp_outputs

def prepare_cp_csv(cp_dict):
    rows = []
    for model_name, data in cp_dict.items():
        lower = data["lower"]
        upper = data["upper"]
        y_true = np.ones_like(lower)

        # coverage per sample
        coverage_per_sample = ((lower <= y_true) & (y_true <= upper)).astype(float)
        coverage_mean = np.mean(coverage_per_sample)
        coverage_ci95 = 1.96 * np.std(coverage_per_sample) / np.sqrt(len(coverage_per_sample))

        # interval width
        width = upper - lower
        width_mean = np.mean(width)
        width_ci95  = 1.96 * np.std(width) / np.sqrt(len(width))

        rows.append({
            "model": model_name,
            "coverage_mean": round(coverage_mean, 2),
            "coverage_ci95": round(coverage_ci95, 2),
            "interval_mean_width": round(width_mean, 2),
            "interval_ci95_width": round(width_ci95, 2)
        })

    return pd.DataFrame(rows)


# #########################################

def main():

    # 1. Carica config
    config = load_config("config_imagenet.yaml")
    eps        = config["eps"]
    batch_size = config["batch_size"]
    norm       = config["norm"]
    device     = config["device"]

    print("\nLoading IMAGENET.")
    
    imagenet_path = "/home/acarlevaro/Sources/albi/Adversarial_CP_V3/InyImageNet/ILSVRC2012_Albi"

    (train, cal, test) = load_imagenet_splits(
        imagenet_path, n_train=config["n_train"], n_cal=config["n_cal"], n_test=config["n_test"]
    )
    
    (train_X, train_y) = train
    (cal_X, cal_y) = cal
    (test_X, test_y) = test

    cal_X  = cal_X.to(device)
    cal_y  = cal_y.to(device)
    test_X = test_X.to(device)
    test_y = test_y.to(device)

    # Lista dove accumulare i DataFrame di tutti i modelli
    all_dfs = []

    # 3. Processa i modelli uno alla volta
    for model_name in config["models"]:
        print(f"\nLoading model: {model_name} ...")
        model = load_rb_model(model_name, 'imagenet', 'cuda',norm = config["corruption"], model_dir=output_dir)
        wrapped = ProbRegressionWrapper(model).to(device).eval()

        # Calibrazione e test clean
        clean_results = extract_scores({model_name: wrapped}, cal_X, cal_y, test_X, test_y, datamode="clean")

        # Genera avversari
        cal_X_adv = attack_with_autoattack(wrapped, cal_X, cal_y, eps=eps, norm=norm, bs=batch_size, device=device)
        test_X_adv = attack_with_autoattack(wrapped, test_X, test_y, eps=eps, norm=norm, bs=batch_size, device=device)

        adv_results = extract_scores({model_name: wrapped}, cal_X_adv, cal_y, test_X_adv, test_y, datamode="adv")

        # Calcola CP
        vanilla_cp_test = compute_cp(clean_results, adv_results, alpha=config["alpha"], cal_on='clean', eval_on='adv', interval_on='test')
        adv_cp_test     = compute_cp(clean_results, adv_results, alpha=config["alpha"], cal_on='adv', eval_on='adv', interval_on='test')

        # Prepara CSV
        df_vanilla = prepare_cp_csv(vanilla_cp_test)
        df_adv     = prepare_cp_csv(adv_cp_test)
        df_vanilla["setting"] = "clean"
        df_adv["setting"]     = "adv"
        df_model = pd.concat([df_vanilla, df_adv], axis=0)
        df_model["model"] = model_name  # assicurati che il nome modello sia presente

        all_dfs.append(df_model)

        # Libera memoria GPU prima del prossimo modello
        del wrapped
        del model
        torch.cuda.empty_cache()

    # Concateno tutti i modelli in un unico CSV
    df_all_models = pd.concat(all_dfs, axis=0)

    # ----------------------------------------------------------------------
    # Salvataggio CSV con numero incrementale
    # ----------------------------------------------------------------------
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    existing = [f for f in os.listdir(results_dir) if f.startswith("cp_results_") and f.endswith(".csv")]
    n = len(existing) + 1
    csv_path = os.path.join(results_dir, f"cp_results_{n}_norm-{norm}_eps-{eps}.csv")
    df_all_models.to_csv(csv_path, index=False)
    print(f"All CP results saved to {csv_path}")

if __name__ == "__main__":
    main()


























