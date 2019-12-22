# Supplemental Materials for ``Universals of word order reflect optimization of grammars for efficient communication''

## Section S1: Formalization of Greenberg Correlation Universals


Here we describe how we selected the word order correlations in Table 1 of the main paper, and how we formalized these using syntactic relations defined by Universal Dependencies.

We base our formalization on the comprehensive study by Dryer \cite{dryer1992greenbergian}.\footnote{Regarding the objections by \citet{dunn2011evolved}, we refer to the follow-ups by \citet{levy2011computational}, and \citet{croft2011greenbergian}.}
Greenberg's original study was based on 30 languages; more recently, Dryer \cite{dryer1992greenbergian} documented the word order correlations based on typological data from 625 languages.
\citet{dryer1992greenbergian} formulated these universals as correlations between the order of objects and verbs and the orders of other syntactic relations.
We test our ordering grammars for these correlations by testing whether the coefficients for these syntactic relations have the same sign as the coefficient of the verb-object relation.
Testing correlations is therefore constrained by the degree to which these relations are annotated in UD.
The verb--object relation corresponds to the  \emph{obj} relation defined by UD.
While most of the other relations also correspond to UD relations, some are not annotated reliably.
We were able formalize eleven out of Dryer's sixteen correlations in UD.
Six of these could not be expressed individually in UD, and were collapsed into three coarse-grained correlations:
First, tense/aspect and negative auxiliaries are together represented by the \emph{aux} relation in UD.
Second, the relation between complementizers and adverbial subordinators with their complement clauses is represented by the \emph{mark} relation.
Third, both the verb-PP relation and the relation between adjectives and their standard of comparison is captured by the \emph{obl} relation.

The resulting operationalization is shown in Table~\ref{table:greenberg-dryer}.
For each relation, we show the direction of the UD syntactic relation: $\rightarrow$ indicates that the verb patterner is the head; $\leftarrow$ indicates that the object patterner is the head.

As described in *Materials and Methods*, we follow \citet{futrell2015largescale} in converting the Universal Dependencies format to a format closer to standard syntactic theory, promoting adpositions, copulas, and complementizers to heads.
As a consequence, the direction of the relations \emph{case}, \emph{cop}, and \emph{mark} is reversed compared to Universal Dependencies.
For clarity, we refer to these reversed relations as \emph{lifted\_case}, \emph{lifted\_cop}, and \emph{lifted\_mark}.


