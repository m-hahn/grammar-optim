# Universals of word order result from optimization of grammars for efficient communication

This repository contains all code and results from the paper.

Code for reproducing statistical analyses and figures is in `results`.
Code for running the neural network models and the control studies reported in SI is in `models`.
Grammar parameters and efficiency scores for all grammars are in `grammars`.

## Requirements
Most analyses only require:
* R: We used version 3.5.1. Analyses require the packages `brms`, `lme4`, `tidyr`, `dplyr`, `ggplot2`.

Creating optimized grammars, or evaluating the efficiency of grammars, requires:

* Python 2.7
* [PyTorch](https://pytorch.org/get-started/locally/), with CUDA. We used PyTorch Version 0.4.1 for experiments, though the code is compatible with more recent versions.
* Extracting real grammars from actual orderings found in corpora additionally requires [Pyro](https://pyro.ai/).
* For the Universal Dependencies corpus data, see `models/corpus_reader/README.md` for instructions.


