
# For each language and each lambda in [0,1], extract the quantile in the baseline distribution
# For each language, find how many baseline languages have better Pareto-optimality

with open("pareto-data-lexicalized.tsv", "r") as inFile:
    data = [x.split(",") for x in inFile.read().strip().replace('"', "").split("\n")]
header = data[0]
header = dict(zip(header, range(len(header))))

data = data[1:]

languages = sorted(set([x[header["Language"]] for x in data]))
print(languages)

import torch

#with open("optimality-by-lambda.tsv", "w") as outFile:
#  print >> outFile, ("\t".join(["Language", "Lambda", "Quantile"]))
for language in languages:
      dataL = [x for x in data if x[header["Language"]] == language]
#      print(language)
      parsGround = float(dataL[0][header["ParsGround"]])
      surpGround = float(dataL[0][header["SurpGround"]])
      models = [x[header["Model"]] for x in dataL]
      surp = [x[header["Surp"]] for x in dataL]
      pars = [x[header["Pars"]] for x in dataL]
      hasNoNA = [x for x in range(len(models)) if surp[x] != "NA" and pars[x] != "NA"]
      models = [models[x] for x in hasNoNA]
      surp = torch.Tensor([float(surp[x]) for x in hasNoNA])
      pars = torch.Tensor([float(pars[x]) for x in hasNoNA])

      # how many baselines Pareto-dominate the real grammar?
      forGround = float(((surp < surpGround) * (pars < parsGround)).float().mean())
#      print(forGround)


      # for each baseline, see how many other baselines Pareto-dominate it
      avgsBetter = 0.0
      for i in range(len(models)):
          avg = ((surp < float(surp[i])) * (pars < float(pars[i]))).float().mean()
          avgsBetter += (1 if forGround < avg else 0)
      print("\t".join([str(x) for x in [language, avgsBetter / len(models)]]))

      # TODO add bootstrapping
