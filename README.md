# Universals of word order result from optimization of grammars for efficient communication

This repository contains all code and results from the paper.

Code for reproducing statistical analyses and figures is in `results`.
Code for running the neural network models and the control studies reported in SI is in `models`.
Grammar parameters and efficiency scores for all grammars are in `grammars`.

## Requirements
Most analyses only require:
* R
Creating optimized grammars, or evaluating the efficiency of grammars, requires:
* Python 2.7
* PyTorch, with CUDA
* corpus data, see `models/corpus_reader/README.md` for instructions.


