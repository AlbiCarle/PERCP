# +
"""
Valutazione PERCP sulle common corruptions: CIFAR10-C, CIFAR100-C, ImageNet-C.
Corruzioni: gaussian_noise, defocus_blur, fog, pixelate. Severita': 1-5.
+ "Worst": per ogni punto, tra le 4 corruzioni (a parita' di severita'),
  tiene quella che degrada di piu' p_true.

------------------------------------------------------------------------
DESIGN (stessa logica delle batterie di attacchi precedenti, adattata):
------------------------------------------------------------------------
- Il regressore (qmodel) e' UNO SOLO, allenato UNA VOLTA su un training set
  PULITO preso dallo split di TRAIN di CIFAR10/100/ImageNet (torchvision).
  Non viene mai riallenato per ogni corruzione/severita': allenarlo 20+
  volte (4 corruzioni x 5 severita') sarebbe uno spreco enorme, dato che
  split conformal non richiede un regressore diverso per ogni condizione -
  la garanzia di coverage viene dalla RICALIBRAZIONE di qhat, non dal
  retraining del modello.

- Per calibrazione e test uso gli split UFFICIALI di RobustBench
  (load_cifar10/load_cifar10c, load_cifar100/load_cifar100c,
  load_imagenet/load_imagenetc): sono allineati indice-per-indice (la
  versione "_c" all'indice i e' la corruzione dell'immagine pulita
  all'indice i), quindi non serve corrompere le immagini a mano - e le
  etichette combaciano automaticamente. Metto comunque un assert di
  sanity-check sulle label, cosi' se l'allineamento dovesse rompersi (per
  una versione diversa della libreria) te ne accorgi subito invece di
  avere risultati silenziosamente sbagliati.

- Questi loader di RobustBench pescano dallo split di TEST/VALIDATION,
  che e' per costruzione disgiunto dal train set usato per il regressore
  -> nessun leakage, meglio ancora della situazione con gli attacchi
  (dove calibrazione e training condividevano la stessa pool di partenza,
  anche se poi splittata correttamente).

- LIMITE NOTO: load_imagenetc restituisce al massimo ~5000 esempi per
  corruzione, indipendentemente da quanti ne chiedi (bug/limite noto della
  libreria, vedi RobustBench issue #92). Tienilo a mente scegliendo n_calib/n_test
  per ImageNet (es. 250+250, non 2000+2000).
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import pandas as pd
import torch.nn.functional as F

from robustbench.data import (
    load_cifar10, load_cifar10c,
    load_cifar100, load_cifar100c,
    load_imagenet, load_imagenetc,
)
from robustbench.loaders import CustomImageFolder
from robustbench.utils import load_model as _rb_load_model
from secml.utils import fm
from secml import settings


def load_imagenetc_fixed(n_examples, severity, data_dir, corruptions, shuffle=False, prepr="none"):
    """Reimplementazione locale di robustbench.data.load_imagenetc.

    Nella versione di RobustBench installata sul sistema c'e' un bug: la
    stringa 'prepr' (es. 'none') viene passata direttamente come transform a
    CustomImageFolder invece di essere prima cercata nel dizionario
    PREPROCESSINGS - causa 'TypeError: str object is not callable' al primo
    __getitem__. Qui il lookup lo faccio a mano, il resto della logica e'
    identico all'originale (stesso percorso data_dir/ImageNet-C/<corruzione>/
    <severita>/<classe>/<img>).
    """
    assert len(corruptions) == 1, "load_imagenetc supporta una sola corruzione per chiamata"
    preprocessings = {
        "Res256Crop224": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()]),
        "Crop288": T.Compose([T.CenterCrop(288), T.ToTensor()]),
        "none": T.Compose([T.ToTensor()]),
    }
    transform = preprocessings[prepr]
    data_folder_path = f"{data_dir}/ImageNet-C/{corruptions[0]}/{severity}"
    imagenet = CustomImageFolder(data_folder_path, transform)
    loader = torch.utils.data.DataLoader(imagenet, batch_size=n_examples, shuffle=shuffle, num_workers=2)
    x_test, y_test, _ = next(iter(loader))
    return x_test, y_test


CORRUPTIONS = ["gaussian_noise", "defocus_blur", "fog", "contrast"] #defocus_blur, fog, pixelate if CIFAR10, CIFAR100 - "motion_blur", "weather/frost", "glass_blur" if IMAGENET
SEVERITIES = [1, 2, 3, 4, 5]

_LOADERS = {
    "CIFAR10": dict(clean=load_cifar10, corrupt=load_cifar10c),
    "CIFAR100": dict(clean=load_cifar100, corrupt=load_cifar100c),
    "IMAGENET": dict(clean=load_imagenet, corrupt=load_imagenetc_fixed),
}


# ----------------------------------------------------------------------
# 1) Modello robusto (threat_model='corruptions' e' quello pensato apposta
#    per questo benchmark; puoi comunque passare 'Linf' se vuoi vedere come
#    se la cava un modello adversarially-robust sulle corruptions, e' una
#    domanda legittima e spesso fatta in letteratura - "robustezza Linf
#    trasferisce alle corruptions?")
# ----------------------------------------------------------------------
def load_corruption_model(model_name, dataset, threat_model="corruptions", device="cuda"):
    output_dir = fm.join(settings.SECML_MODELS_DIR, "robustbench")
    model = _rb_load_model(
        model_name=model_name, dataset=dataset.lower(),
        threat_model=threat_model, model_dir=output_dir,
    )
    model.eval()
    model.to(device)
    return model


# ----------------------------------------------------------------------
# 2) Training set PULITO per il regressore (split train, mai toccato dai
#    loader di corruzione che invece pescano dal test/val set)
# ----------------------------------------------------------------------
def load_training_set(dataset_name, root, imagenet_train_dir=None):
    dataset_name = dataset_name.upper()
    if dataset_name == "CIFAR10":
        return torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=T.ToTensor())
    if dataset_name == "CIFAR100":
        return torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=T.ToTensor())
    if dataset_name == "IMAGENET":
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        return torchvision.datasets.ImageFolder(imagenet_train_dir or root, transform=transform)
    raise ValueError(f"Dataset {dataset_name} non supportato.")


def load_imagenet_training_fallback(n_train, data_dir, corruption="jpeg_compression", severity=1):
    """Ripiego per allenare il regressore su ImageNet quando la cartella
    'val/' pulita (richiesta da load_imagenet) non e' disponibile/strutturata
    correttamente - vedi discussione precedente sul FileNotFoundError.

    ATTENZIONE - compromesso metodologico, non l'ideale:
    - non sono immagini davvero pulite, solo blandamente corrotte
      (severity=1)
    - uso di default 'jpeg_compression', che NON fa parte della batteria
      di valutazione (gaussian_noise, defocus_blur, fog, pixelate) - se lo
      cambi, assicurati che resti DIVERSO da quelli, altrimenti stai
      allenando il regressore proprio sulla condizione che poi valuti,
      falsando il confronto vanilla/PERCP a favore di PERCP.
    - UNA SOLA chiamata (non due identiche): qui serve solo un pool di
      training, non una coppia calib/test.

    La soluzione corretta resta sistemare 'val/' e usare load_imagenet vero.
    """
    X_train, y_train = load_imagenetc(
        n_examples=n_train, corruptions=[corruption], severity=severity, data_dir=data_dir
    )
    return X_train, y_train


# ----------------------------------------------------------------------
# 3) Coppie clean/corrotto allineate, per una data corruzione+severita'
# ----------------------------------------------------------------------
def get_clean_corrupt_pair(dataset_name, corruption, severity, n_calib, n_test,
                            data_dir_clean, data_dir_corrupt=None, seed=None):
    """seed=None (default): prende le prime n_calib+n_test immagini del file
    di quella corruzione+severita' (deterministico, shuffle=False lato
    RobustBench) e le affetta in due blocchi disgiunti [:n_calib] / [n_calib:].

    seed=<int>: oltre a prendere le stesse n_calib+n_test immagini, applica
    UNA permutazione casuale (seedata) IDENTICA sia alla versione pulita che
    a quella corrotta prima di affettare - necessario per randomizzare il
    campionamento senza rompere l'allineamento indice-per-indice tra le due
    (se permutassi le due chiamate indipendentemente, l'immagine i di
    X_clean e l'immagine i di X_corr non sarebbero piu' la stessa foto).

    data_dir_clean / data_dir_corrupt: per CIFAR di solito coincidono (stessa
    cartella 'data' per entrambi); per IMAGENET tipicamente NO - clean e
    ImageNet-C spesso vivono in posti diversi sul disco, quindi qui sono due
    parametri separati (se data_dir_corrupt non e' dato, riuso data_dir_clean).
    NOTA sui path attesi da RobustBench:
    - clean (load_imagenet): data_dir_clean/val/<classe>/<img>
    - corrotto (load_imagenetc): data_dir_corrupt/ImageNet-C/<corruzione>/
      <severita>/<classe>/<img> - la funzione aggiunge "ImageNet-C" da sola,
      quindi data_dir_corrupt va passato SENZA "/ImageNet-C" in fondo.
    """
    dataset_name = dataset_name.upper()
    loaders = _LOADERS[dataset_name]
    n_total = n_calib + n_test
    if data_dir_corrupt is None:
        data_dir_corrupt = data_dir_clean

    X_clean_all, y_all = loaders["clean"](n_examples=n_total, data_dir=data_dir_clean)

    corrupt_kwargs = dict(n_examples=n_total, corruptions=[corruption], severity=severity,
                           data_dir=data_dir_corrupt, shuffle=False)
    if dataset_name == "IMAGENET":
        # le immagini di ImageNet-C sono gia' 224x224: il resize+crop di default
        # ('Res256Crop224') le altera inutilmente - vedi RobustBench issue #59,
        # 'prepr=none' e' quello che il team stesso raccomanda.
        corrupt_kwargs["prepr"] = "none"
    X_corr_all, y_corr_all = loaders["corrupt"](**corrupt_kwargs)

    assert torch.equal(y_all[:n_total], y_corr_all[:n_total]), (
        "Le label di clean e corrotto non coincidono: probabile disallineamento "
        "tra load_*() e load_*c(). Verifica shuffle=False su entrambi e la "
        "versione di robustbench installata."
    )

    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=g)
        X_clean_all, y_all = X_clean_all[perm], y_all[perm]
        X_corr_all = X_corr_all[perm]

    X_calib_clean, y_calib = X_clean_all[:n_calib], y_all[:n_calib]
    X_test_clean, y_test = X_clean_all[n_calib:n_total], y_all[n_calib:n_total]
    X_calib_corr = X_corr_all[:n_calib]
    X_test_corr = X_corr_all[n_calib:n_total]

    return X_calib_clean, y_calib, X_calib_corr, X_test_clean, y_test, X_test_corr


# ----------------------------------------------------------------------
# 4) Training del regressore - UNA VOLTA SOLA
# ----------------------------------------------------------------------
def train_percp_regressor(model_rb, QuantileRegressor, ProbabilityRegressionDataset,
                           cqr_loss, get_true_probabilities,
                           X_train, y_train, alpha=0.1, epochs=50, device="cuda"):
    p_train = get_true_probabilities(model_rb, X_train, y_train).view(-1)
    train_loader = torch.utils.data.DataLoader(
        ProbabilityRegressionDataset(X_train, p_train.cpu()), batch_size=32, shuffle=True
    )
    qmodel = QuantileRegressor().to(device)
    optimizer = torch.optim.Adam(qmodel.parameters(), lr=1e-4)
    for epoch in range(epochs):
        qmodel.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = qmodel(Xb)
            loss = cqr_loss(pred, yb, alpha)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Regressor] Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    qmodel.eval()
    return qmodel


# ----------------------------------------------------------------------
# 5) Valutazione PERCP per UNA coppia clean/corrotto gia' pronta
# ----------------------------------------------------------------------
def evaluate_corruption_percp(qmodel, model_rb, get_true_probabilities,
                               X_calib_clean, y_calib, X_calib_corr,
                               X_test_clean, y_test, X_test_corr,
                               alpha=0.1, device="cuda"):
    qmodel.eval()

    p_calib_clean = get_true_probabilities(model_rb, X_calib_clean, y_calib).to(device).view(-1)
    p_calib_corr = get_true_probabilities(model_rb, X_calib_corr, y_calib).to(device).view(-1)
    p_test_corr = get_true_probabilities(model_rb, X_test_corr, y_test).to(device).view(-1)

    with torch.no_grad():
        pred_calib_clean = qmodel(X_calib_clean.to(device))
        pred_calib_corr = qmodel(X_calib_corr.to(device))
        pred_test_corr = qmodel(X_test_corr.to(device))

    # VANILLA: qhat da calibrazione pulita
    scores_clean = torch.maximum(pred_calib_clean[:, 0] - p_calib_clean,
                                  p_calib_clean - pred_calib_clean[:, 1])
    qhat_vanilla = torch.quantile(scores_clean, 1 - alpha)
    lo_v = pred_test_corr[:, 0] - qhat_vanilla
    hi_v = pred_test_corr[:, 1] + qhat_vanilla
    coverage_v = ((p_test_corr >= lo_v) & (p_test_corr <= hi_v)).float().mean()
    size_v = (hi_v - lo_v).mean()

    # PERCP: qhat ricalibrato su calibrazione CORROTTA (stessa corruzione/severita' del test)
    scores_corr = torch.maximum(pred_calib_corr[:, 0] - p_calib_corr,
                                 p_calib_corr - pred_calib_corr[:, 1])
    qhat_percp = torch.quantile(scores_corr, 1 - alpha)
    lo_p = pred_test_corr[:, 0] - qhat_percp
    hi_p = pred_test_corr[:, 1] + qhat_percp
    coverage_p = ((p_test_corr >= lo_p) & (p_test_corr <= hi_p)).float().mean()
    size_p = (hi_p - lo_p).mean()

    return {
        "coverage_vanilla": coverage_v.item(), "size_vanilla": size_v.item(),
        "coverage_percp": coverage_p.item(), "size_percp": size_p.item(),
        "qhat_vanilla": qhat_vanilla.item(), "qhat_percp": qhat_percp.item(),
    }


# ----------------------------------------------------------------------
# 6) "Worst" tra le corruzioni, a parita' di severita'
# ----------------------------------------------------------------------
def evaluate_corruption_worst(qmodel, model_rb, get_true_probabilities, pairs,
                               alpha=0.1, device="cuda"):
    """pairs: dict {corruption_name: (X_calib_clean, y_calib, X_calib_corr,
    X_test_clean, y_test, X_test_corr)} per la STESSA severita'. X_calib_clean/
    y_calib/X_test_clean/y_test sono identici in tutte le entry (stessi punti,
    cambia solo la corruzione applicata) - li prendo dalla prima.
    """
    X_calib_clean, y_calib, _, X_test_clean, y_test, _ = next(iter(pairs.values()))

    calib_candidates, calib_p, test_candidates, test_p = [], [], [], []
    for name, (_, _, X_calib_corr, _, _, X_test_corr) in pairs.items():
        calib_candidates.append(X_calib_corr)
        calib_p.append(get_true_probabilities(model_rb, X_calib_corr, y_calib).view(-1).cpu())
        test_candidates.append(X_test_corr)
        test_p.append(get_true_probabilities(model_rb, X_test_corr, y_test).view(-1).cpu())

    def _pick_worst(candidates, p_list):
        P = torch.stack(p_list, dim=0)          # (n_corruptions, N)
        idx = P.argmin(dim=0)                     # (N,) corruzione peggiore per punto
        stack = torch.stack(candidates, dim=0)     # (n_corruptions, N, C, H, W)
        N = stack.shape[1]
        return stack[idx, torch.arange(N)]

    X_calib_worst = _pick_worst(calib_candidates, calib_p)
    X_test_worst = _pick_worst(test_candidates, test_p)

    return evaluate_corruption_percp(
        qmodel, model_rb, get_true_probabilities,
        X_calib_clean, y_calib, X_calib_worst,
        X_test_clean, y_test, X_test_worst,
        alpha=alpha, device=device,
    )


# ----------------------------------------------------------------------
# 7) Orchestratore: tutta la griglia corruzioni x severita' + Worst
# ----------------------------------------------------------------------
def run_corruption_battery(dataset_name, model_rb, qmodel, get_true_probabilities,
                            n_calib=500, n_test=500, data_dir_clean="./data", data_dir_corrupt=None,
                            corruptions=CORRUPTIONS, severities=SEVERITIES,
                            alpha=0.1, device="cuda", seed=None):
    """seed: se dato, randomizza il campionamento (vedi get_clean_corrupt_pair).
    Uso LO STESSO seed per tutte le corruzioni di una data severita', perche'
    "Worst" confronta le corruzioni punto per punto: se ognuna pescasse un
    sottoinsieme diverso di immagini, il confronto non avrebbe piu' senso.

    data_dir_clean / data_dir_corrupt: per CIFAR di solito coincidono; per
    IMAGENET tipicamente sono due cartelle diverse (vedi get_clean_corrupt_pair).
    """
    rows = []
    for severity in severities:
        print(f"\n=== Severity {severity} ===")
        pairs = {}
        for corruption in corruptions:
            print(f"--- {corruption} (severity {severity}) ---")
            pair = get_clean_corrupt_pair(dataset_name, corruption, severity, n_calib, n_test,
                                           data_dir_clean, data_dir_corrupt=data_dir_corrupt, seed=seed)
            pairs[corruption] = pair
            X_calib_clean, y_calib, X_calib_corr, X_test_clean, y_test, X_test_corr = pair
            res = evaluate_corruption_percp(
                qmodel, model_rb, get_true_probabilities,
                X_calib_clean, y_calib, X_calib_corr,
                X_test_clean, y_test, X_test_corr,
                alpha=alpha, device=device,
            )
            res.update({"dataset": dataset_name, "corruption": corruption, "severity": severity})
            rows.append(res)

        print(f"--- WORST (severity {severity}) ---")
        res_worst = evaluate_corruption_worst(qmodel, model_rb, get_true_probabilities, pairs,
                                               alpha=alpha, device=device)
        res_worst.update({"dataset": dataset_name, "corruption": "WORST", "severity": severity})
        rows.append(res_worst)

    df = pd.DataFrame(rows)
    df["coverage_vanilla"] *= 100
    df["coverage_percp"] *= 100
    return df


# +
class RobustBenchProbabilityRegressor(nn.Module):

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier


    def forward(self, x, y):

        logits = self.classifier(x)

        probs = F.softmax(logits, dim=1)

        idx = torch.arange(
            len(y),
            device=x.device
        )

        p_true = probs[idx,y]

        return p_true
    
def get_probability_targets(model,dataset,indices):

    X=[]
    y=[]
    p=[]


    with torch.no_grad():

        for i in indices:

            img,label=dataset[i]

            img=img.unsqueeze(0).cuda()
            label=torch.tensor(
                [label],
                device="cuda"
            )


            prob=model(
                img,
                label
            )


            X.append(img.cpu())
            y.append(label.cpu())
            p.append(prob.cpu())


    return (
        torch.cat(X),
        torch.cat(y),
        torch.cat(p)
    )

def get_true_probabilities(
    classifier,
    X,
    y,
    device="cuda"
):

    classifier.eval()

    probs_all=[]

    batch_size=64

    with torch.no_grad():

        for i in range(0,len(X),batch_size):

            x=X[i:i+batch_size].to(device)
            labels=y[i:i+batch_size].to(device)

            logits=classifier(x)

            probs=torch.softmax(
                logits,
                dim=1
            )

            p_true=probs[
                torch.arange(len(labels),device=device),
                labels
            ]

            probs_all.append(
                p_true.cpu()
            )

    return torch.cat(probs_all)

def split_train_calib(X_cal, y_cal, frac_train=0.5, seed=None):
    """Divide il pool 'X_cal' in due parti disgiunte:
    - X_train/y_train: usati per fittare il quantile regressor
    - X_calib/y_calib: usati SOLO per calcolare qhat (mai visti in training)
    """
    if seed is not None:
        torch.manual_seed(seed)
 
    n = len(X_cal)
    perm = torch.randperm(n)
    n_train = int(n * frac_train)
 
    idx_train = perm[:n_train]
    idx_calib = perm[n_train:]
 
    return (
        X_cal[idx_train], y_cal[idx_train],
        X_cal[idx_calib], y_cal[idx_calib],
    )
 

from torch.utils.data import Dataset, DataLoader


class ProbabilityRegressionDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx]
        )
    
import torchvision.models as models


class QuantileRegressor(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )


    def forward(self,x):

        out = self.model(x)

        return out
    
def pinball_loss(pred, target, quantile):

    error = target - pred

    return torch.mean(
        torch.maximum(
            quantile*error,
            (quantile-1)*error
        )
    )


def cqr_loss(pred, target, alpha):
    
    q_low=alpha/2
    q_high=1-alpha/2

    loss_low = pinball_loss(
        pred[:,0],
        target,
        q_low
    )

    loss_high = pinball_loss(
        pred[:,1],
        target,
        q_high
    )

    return loss_low + loss_high    


# +
DATASET = "IMAGENET"     # "CIFAR10", "CIFAR100", "IMAGENET"

# --- IMAGENET: clean e corrotto vivono in due posti diversi sul disco ---
# clean: serve un sottolivello 'val/' che nel tuo caso non c'e' -> symlink:
#   ln -s /home/acarlevaro/Sources/albi/OLD/Adversarial_CP_V3/InyImageNet/ILSVRC2012_Albi \
#         /home/acarlevaro/Sources/albi/OLD/Adversarial_CP_V3/InyImageNet/val
DATA_DIR_CLEAN = "/home/acarlevaro/Sources/albi/OLD/Adversarial_CP_V3/InyImageNet"

# corrotto: data_dir va SENZA '/ImageNet-C' finale, lo aggiunge load_imagenetc da solo
DATA_DIR_CORRUPT = "/home/acarlevaro/Sources/albi/OLD/Extension/AttackBench/IMAGENET/data/"

model_rb = load_corruption_model(
    model_name="Alexnet",         # o un modello dalla leaderboard 'corruptions' per il tuo dataset
    dataset=DATASET,
    threat_model="corruptions",
)

# --- training set pulito per il regressore ---
if DATASET == "IMAGENET":
    train_dataset = load_training_set(DATASET, root=None, imagenet_train_dir=DATA_DIR_CLEAN + "/val")
else:
    train_dataset = load_training_set(DATASET, root="/home/acarlevaro/Sources/albi/data")

regressor = RobustBenchProbabilityRegressor(model_rb).cuda()   # gia' nel tuo notebook
idx_train = torch.randperm(len(train_dataset))[:300]
X_train, y_train, _ = get_probability_targets(regressor, train_dataset, idx_train)

qmodel = train_percp_regressor(
    model_rb=model_rb,
    QuantileRegressor=QuantileRegressor,
    ProbabilityRegressionDataset=ProbabilityRegressionDataset,
    cqr_loss=cqr_loss,
    get_true_probabilities=get_true_probabilities,
    X_train=X_train, y_train=y_train,
    alpha=0.1, epochs=50,
)

df = run_corruption_battery(
    dataset_name=DATASET,
    model_rb=model_rb,
    qmodel=qmodel,
    get_true_probabilities=get_true_probabilities,
    n_calib=250, n_test=250,     # cap ~5000 di load_imagenetc, stai ampiamente dentro
    data_dir_clean=DATA_DIR_CLEAN,
    data_dir_corrupt=DATA_DIR_CORRUPT,
    alpha=0.1,
)
df
# -


