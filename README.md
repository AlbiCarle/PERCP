# Provably Efficient and Robust Conformal Prediction under a Realistic Threat Model (submitted to COPA25)

[![Built with SECML](https://img.shields.io/badge/Built%20with-SECML-4c1.svg)](https://secml.readthedocs.io/en/v0.15/)
[![Uses RobustBench](https://img.shields.io/badge/Uses-RobustBench-blue)](https://robustbench.github.io/)

**Alberto Carlevaro<sup>1,2</sup>**, **Luca Oneto<sup>2</sup>**, **Davide Anguita<sup>2</sup>**, and **Fabio Roli<sup>2,3</sup>**
1. Aitek S.p.A., Funded Research Department, Via della Crocetta 15, 16122 Genova, Italy.
2. CNR - Istituto di Elettronica e di Ingegneria dell’Informazione e delle Telecomunicazioni (CNR-IEIIT)
2. Dipartimento di Informatica, Bioingegneria, Robotica e Ingegneria dei Sistemi (DIBRIS), University of Genoa, Italy  
3. DIEE, University of Cagliari, Via Marengo, Cagliari, 09123, Italy.
---

## Abstract

Robust conformal prediction is a model-agnostic technique designed to construct predictive sets with guaranteed coverage, assuming data exchangeability, even under adversarial
attacks. Two primary strategies have been explored to address vulnerabilities to these
attacks. The first strategy employs randomization, which is computationally efficient but
fails to provide formal performance guarantees without resulting in overly conservative
predictive sets. The second strategy involves formal verification, which restores coverage
guarantees but leads to excessively conservative predictive sets and prohibitive computational overhead. Indeed, verification generally becomes NP-hard as it attempts to cope
with attacks that are practically impossible, rendering some security claims unfalsifiable.
In this paper, we propose a novel, provably efficient robust conformal prediction method
by clearly defining a realistic threat model. Specifically, we assume explicit knowledge of
the set of potential adversarial attacks, aligning our approach with standard certification
procedures designed to certify against specific, identified threats. We demonstrate that
attacks targeting the model can effectively be reframed as attacks on the score function,
allowing us to recalibrate the score quantile to account for these known attacks and thereby
restore desired coverage guarantees. It is worth noting that our approach allows to easily
incorporate unknown or emerging (zero-day) attacks upon discovery, thus reestablishing
coverage guarantees. By avoiding computationally intensive verification and operating
under realistic threat assumptions, our approach achieves both efficiency and provable robustness. Empirical evaluations on real-world classification datasets and comparisons with
state-of-the-art methods support the effectiveness and practicality of our proposed solution.

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

## How to cite this work

Still under submission

---
