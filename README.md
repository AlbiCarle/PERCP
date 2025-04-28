# Provably and Efficient Robust Conformal Prediction with a Realistic Threat Model (submitted to COPA25)

[![arXiv](https://img.shields.io/badge/arXiv-2405.18942-b31b1b.svg)](https://arxiv.org/abs/2405.18942)
[![Built with SECML](https://img.shields.io/badge/Built%20with-SECML-4c1.svg)](https://secml.readthedocs.io/en/v0.15/)

**Alberto Carlevaro^1**, **Luca Oneto^2**, **Davide Anguita^2**, and **Fabio Roli^2**  
1. CNR - Istituto di Elettronica e di Ingegneria dell’Informazione e delle Telecomunicazioni (CNR-IEIIT)
2. Dipartimento di Informatica, Bioingegneria, Robotica e Ingegneria dei Sistemi (DIBRIS), University of Genoa, Italy  

---

## Table of Contents

- [Abstract](#abstract)
- [How to Use](#how-to-use)
- [Libraries Used](#libraries-used)
- [Future Steps](#future-steps)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Abstract

Robust conformal prediction is a model-agnostic technique for constructing predictive sets with guaranteed coverage under the assumption of data exchangeability, even in the presence of adversarial attacks.  
Two primary strategies have been explored to address vulnerabilities to these attacks: randomization (efficient but overly conservative) and formal verification (provably robust but computationally prohibitive and often unfalsifiable).

In this paper, we propose a novel, provably efficient robust conformal prediction method by defining a realistic threat model: we assume explicit knowledge of the set of potential adversarial attacks, aligning with standard certification practices. We demonstrate that model attacks can be reframed as attacks on the score function, allowing recalibration of the score quantile to restore coverage guarantees without heavy verification.

Although initially limited to known attacks, the framework can rapidly incorporate zero-day attacks as they are discovered, maintaining robustness.  
Our empirical evaluation on real-world classification datasets shows that our approach achieves strong practical performance compared to state-of-the-art methods, with significant computational efficiency.

---

## How to Use

The code is organized into three separate codebases, reflecting the three main experimental case studies discussed in the paper.

Each dataset folder contains two Jupyter Notebooks:

- **Comparison.ipynb**:  
  Run experiments comparing **PERCP** with existing state-of-the-art methods, including randomized smoothing and verification algorithms.  
  We adapted code from [Verifiably Robust Conformal Prediction](https://github.com/ddv-lab/Verifiably_Robust_CP) ([paper link](https://arxiv.org/pdf/2405.18942)) by Jeary *et al.*.  
  This notebook replicates **Table 1** and **Figure 1** of the paper.

- **Benchmarks.ipynb**:  
  Run experiments testing **PERCP** on robust classification models from [RobustBench](https://robustbench.github.io/).  
  This notebook replicates **Table 2** of the paper.

---

## Libraries Used

- **SECML** ([Documentation](https://secml.readthedocs.io/en/v0.15/)) was used to implement and run adversarial attacks, including:
  - Projected Gradient Descent (PGD)
  - Fast Gradient Sign Method (FGSM)
  - Carlini & Wagner (CW)
  - DeepFool
  - Basic Iterative Method (BIM)

Please make sure you install [SECML](https://secml.readthedocs.io/en/v0.15/) and other requirements before running the notebooks.

---

## Future Steps

- Extend **PERCP** to regression problems.
- Incorporate more diverse and stronger adversarial attacks.
- Investigate adaptive strategies for unknown (zero-day) attacks.

---

## Contact

For any questions regarding the work or the code in this repository, feel free to contact:  
📧 **Alberto Carlevaro** — albertocarlevaro@gmail.com

---

## Acknowledgements

This work was partially supported by:

- Project **SERICS** (PE00000014) under the NRRP MUR program funded by the EU - NGEU.
- Project **FAIR** (PE00000013) under the NRRP MUR program funded by the EU - NGEU.
- **REXASI-PRO** H-EU project (HORIZON-CL4-2021-HUMAN-01-01, Grant Agreement ID: 101070028).
- **Fit4MedRob - Fit for Medical Robotics** Grant (PNC0000007).
- **ELSA – European Lighthouse on Secure and Safe AI** funded by the European Union’s Horizon Europe (Grant Agreement No. 101070617).

---
