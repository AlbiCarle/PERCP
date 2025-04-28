# Provably and Efficient Robust Conformal Prediction with a Realistic Threat Model (submitted to COPA25)

Alberto Carlevaro\*, Luca Oneto\*, Davide Anguita and Fabio Roli.

Dipartimento delle scienze e tecnologie Informatiche, Bioingegneria, Robotica e Ingegneria dei Sistemi (DIBRIS), University of Genoa, Italy.
CNR-Istituto di Elettronica e di Ingegneria dell’Informazione e delle Telecomunicazioni (CNR-IEIIT).

## Abstract

Robust conformal prediction is a model-agnostic technique designed to construct predic-
tive sets with guaranteed coverage, assuming data exchangeability, even under adversarial
attacks. Two primary strategies have been explored to address vulnerabilities to these
attacks. The first strategy employs randomization, which is computationally efficient but
fails to provide formal performance guarantees without resulting in overly conservative
predictive sets. The second strategy involves formal verification, which restores coverage
guarantees but leads to excessively conservative predictive sets and prohibitive computa-
tional overhead. Indeed, verification generally becomes NP-hard as it attempts to cope
with attacks that are practically impossible, rendering some security claims unfalsifiable.
In this paper, we propose a novel, provably efficient robust conformal prediction method
by clearly defining a realistic threat model. Specifically, we assume explicit knowledge of
the set of potential adversarial attacks, aligning our approach with standard certification
procedures designed to certify against specific, identified threats. We demonstrate that
attacks targeting the model can effectively be reframed as attacks on the score function,
allowing us to recalibrate the score quantile to account for these known attacks and thereby
restore desired coverage guarantees. While our initial approach does not cover unknown
or emerging (zero-day) attacks, such threats can be swiftly incorporated (patched) into
our framework upon discovery, thus reestablishing coverage guarantees. By avoiding com-
putationally intensive verification and operating under realistic threat assumptions, our
approach achieves both efficiency and provable robustness. Empirical evaluations on real-
world classification datasets and comparisons with state-of-the-art methods substantiate
the effectiveness and practicality of our proposed solution.
Keywords: Robust Conformal Prediction, Computational Efficiency, Provable Guaran-
tees, Unfalsifiability, Realistic Threat Models.

## How to use

The code (currently) exists as two separate codebases which directly reflect the two main sets of experiments and their case studies that exist in the paper.

To run either the classification or regression experiments, the full steps and setup instructions can be found in the corresponding INSTRUCTIONS.md files in each folder. 

## Future Steps

The code currently exists so as to replicate the experiments in the paper, however work has already begun on wrapping up both classification and regression codebases into a single, modular and configurable library for general use.

It is anticipated that this work will be completed by Q2 2025. Please continue to check back for updates.

## Contact

If you have any questions regarding the work or the code in this repository, please contact Tom Kuipers in the first instance (email: first.lastname (at) kcl.ac.uk).

## Acknowledgements

This work is supported by the “REXASI-PRO” H-EU project, call HORIZON-CL4-2021-HUMAN-01-01, Grant agreement ID: 101070028.
