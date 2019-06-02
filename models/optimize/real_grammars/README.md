# Extracting Real Grammars for Corpora

* `inferWeightsCrossVariationalAllCorpora_NoPunct_NEWPYTORCH_FuncHead_Coarse.py`: Extracts a grammar from the corpus of a language.
* `createWeightsModelsPerLanguage_FuncHead_Coarse.py`: Wrapper that extracts a grammar for each corpus.

Auxiliary:
* `filterWeights_Coarse.py`: In case grammars were extracted from the same language multiple times due to parallelization, selects one of these files.


