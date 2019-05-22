with open("efficiency-results-lme4.tsv", "r") as inFile:
   lme4 = [x.strip().split("\t") for x in (inFile).read().strip().split("\n")]
with open("output/results-prevalence-two-09-large.tsv", "r") as inFile:
   brms = [x.strip().split("\t") for x in (inFile).read().strip().split("\n")][1:]
dependencies = ["acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp"]

brms = [x for x in brms if x[0] in dependencies]

with open("output/efficiency-results-full.tex", "w") as outFile:
 for lme, brm in zip(lme4, brms):
   brm[0] = brm[0].replace("_", "\_")
   line = brm + lme[1:]
   if line[0] == "aux":
      for i in [1,4]:
         line[i] = 1-float(line[i])
      for i in [2,5,7]:
         line[i] = -float(line[i])

   for i in [1,2,3,5,6,7]:
      line[i] = str(round(float(line[i]),3))
   for j in [4, 8]:
      line[j] = float(line[j])
      if line[j] == 0:
        line[j] = "$<$ \\num{1e-4}"
#      else:
      else:
         if round(line[j], 3) == 0:
           line[j] = "\\num{" + '{:0.1e}'.format(line[j]) + "}"
         else:
             line[j] = round(line[j], 3)
         line[j] = str(line[j])
   # acl & 0.793367346938776 & 1.50549627266397 & 0.326852828127335 & 0 & 1.434444 & 0.2716449 & 5.280586 & 1.287717e-07

   print((" & ".join(line)) + "\\\\", file=outFile)



