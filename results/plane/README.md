# Visualizing the cost-informativity plane (Study 1).

Workflow:

* Study 1:
    + Run `visualizePlane_Averaged.R` for figure in main paper.
    * Run `optimalityTests_perLanguage.R` to evaluate how many real grammars improve over baselines.
* Study 1, SI:
    * Run `visualizePlane_perLanguage.R`, `visualizePlane_perLanguage_untransformed.R` for per-language result figure
    * Run `recordBaselinePositions.R` to prepare analyses of joint optimization across different values of lambda, then run scripts in `analyze_pareto_optimality/`.
* Other analyses:
    * Run `visualizePlane_acrossLanguages.R` to visualize plot showing that grammars are optimized specifically for the tree structures of the language.

See `nondeterministic/`, `pureUD/`, `unlexicalized/` for control studies.


