# Grammar Parameters and Efficiency Values

Efficiency Values:

* `plane`

Grammar Parameters (Main Experiments):

* `manual_output_funchead_ground_coarse_final`: parameters for the real grammars extracted from actual orderings
* `manual_output_funchead_RANDOM{,2,3,4,5}`: parameters for the 50 baseline grammars for each of the 51 languages
* `manual_output_funchead_two_coarse_lambda09_best_large`: parameters for grammars optimized for efficiency

Grammar Parameters (Other Experiments):

* `manual_output_funchead_coarse_depl`: Grammars optimized for dependency length (Section S10)
* `manual_output_funchead_langmod_coarse_best_balanced`: Grammars optimized for predictability (Section S4)
* `manual_output_funchead_two_coarse_parser_best_balanced`: Grammars optimized for predictability (Section S4)
* `manual_output_funchead_two_coarse_lambda09_best_balanced`: Previous preregistered experiment, reported in Section S4.6 (http://aspredicted.org/blind.php?x=8gp2bt)
* `manual_output_funchead_two_coarse_final`: Previous preregistered experiment, reported in Section S4.6 (https://aspredicted.org/blind.php?x=bg35x7)
* `parser-coarse-balanced`: Hyperparameter search (Section S6)
* `pred-coarse-balanced`: Hyperparameter search (Section S6)
* `pred-parser-coarse`: Hyperparameter search (Section S6)
* `pred-parser-coarse-lambda09-balanced`: Hyperparameter search (Section S6)

Additional Data:
* `dependency_length`: dependency length estimates for grammars (Section S10)

