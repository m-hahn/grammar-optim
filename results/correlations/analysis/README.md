# Bayesian Logistic Regression

Main analysis:
* `evaluation-efficiency.R` runs the Bayesian regression for the eight correlations and stores the posterior samples

Other analyses, reported in SI:

* `evaluation-{efficiency, depl, pred, pars}.R` runs the Bayesian regression for the eight correlations and stores the posterior samples (Section S4.4)
* `evaluation-{efficiency, depl, pred, pars}-perRelation.R` runs an individual Bayesian regression for every UD relation and stores the posterior samples (Section S4.5)
* `efficiency-control-brms.R` and `efficiency-control-lme4.R` report numerical results of Bayesian analysis and corresponding frequentist control (Section S4.3)
* `evaluation-efficiency-prior09.R`, `evaluation-efficiency-prior10.R` runs analysis on our previous preregistered experiments (Section S4.6)

