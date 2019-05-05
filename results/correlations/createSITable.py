with open("../relations.tsv", "r") as inFile:
   rels = [x.strip() for x in inFile.read().strip().split("\n")]
for x in rels:
   print(x+"  &  \includegraphics[width=0.06\textwidth]{../results/correlations/figures/posteriors/posterior_perRelation_Real_"+x+".pdf}   &   \includegraphics[width=0.06\textwidth]{../results/correlations/figures/posteriors/posterior_perRelation_DependencyLength_"+x+".pdf}   &   \includegraphics[width=0.06\textwidth]{../results/correlations/figures/posteriors/posterior_perRelation_Predictability_"+x+".pdf}  &   \includegraphics[width=0.06\textwidth]{../results/correlations/figures/posteriors/posterior_perRelation_Parseability_"+x+".pdf}  &  \includegraphics[width=0.06\textwidth]{../results/correlations/figures/posteriors/posterior_perRelation_Efficiency_"+x+".pdf}   & ")

