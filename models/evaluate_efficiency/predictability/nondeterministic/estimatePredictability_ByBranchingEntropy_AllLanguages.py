import os
import sys

BASE_DIR = sys.argv[1]

# This is for cases where only the per-language optimal models require calculation of predictability.
if len(sys.argv) > 2:
   listPath = sys.argv[2] # e.g. best-parse-best-balanced.csv
else:
   listPath = None

import os

import subprocess
import random

from math import exp, sqrt
import sys

modelNumbers = None
languages = None

if listPath is not None:
 with open(listPath, "r") as inFiles:
   relevantModels = [x.split(",") for x in inFiles.read().strip().replace('"',"").split("\n")]
   modelIndex = relevantModels[0].index("Model")
   typeIndex = relevantModels[0].index("Type")
   languageIndex = relevantModels[0].index("Language")

   assert all([x[typeIndex] == BASE_DIR for x in relevantModels[1:]]), [x[typeIndex] for x in relevantModels[1:]]
   relevantModels = [(x[languageIndex], x[modelIndex]) for x in relevantModels[1:]]

models =[x for x in  os.listdir("../../../../raw-results/"+BASE_DIR+"/") if x.endswith(".tsv")]
modelsProcessed = []
for i in range(len(models)):
   print(BASE_DIR, models[i])
   if models[i] == "auto-summary-lstm.tsv":
      continue
   if "ground" in BASE_DIR:
      ind = models[i].index("_inferWe")
      language = models[i][:ind]
      number = models[i][:-4].split("_")[-1]
   elif "RANDOM" in BASE_DIR:
      mod = models[i].split("_")
      language = models[i][:models[i].index("_RANDOM")]
      number = mod[-1][:-4]
   else:
      if "_readData" not in models[i]:
         continue
      ind = models[i].index("_readData")
      language = models[i][:ind]
      number = models[i][:-4].split("_")[-1]
   if modelNumbers is not None and number not in modelNumbers:
      continue
   models[i] = (language, number)
   modelsProcessed.append(models[i])
models = modelsProcessed
assert len(models) > 0
print(models)
print(modelNumbers)
models = [model for model in models if len(model) == 2]

if listPath is None:
    relevantModels = models

import os


scripts = ["estimatePredictability_ByBranchingEntropy.py"]



failures = 0

while failures < 20000:
  existingFiles = os.listdir("../../../../raw-results//language_modeling_coarse_plane_fixed_nondeterministic")
  script = random.choice(scripts) #scripts[0] if random.random() < 0.8 else scripts[1]
  language, model = random.choice(relevantModels)
  if languages is not None and language not in languages:
      continue
  if language not in ["English", "Japanese", "Czech"]:
     failures += 1
     continue
  existing = [x for x in existingFiles if x.startswith(language) and "_"+model+"_" in x]
  if len(existing) > 0: #random.random() > ((1.0/(1+len(existing)))):
     print(existing)
     print("Language model for this model exists "+str(((1.0/(1+len(existing))))))
     failures += 1
     continue
  failures = 0
#  existing = [x for x in existingFiles if x.startswith(language)]
#  if len(existing) > 5:
#    if random.random() > 0.3:
#        print("Skipping "+language)
#        continue
  entropy_weight = random.choice([0.001, 0.001,0.001, 0.001,  0.01, 0.1, 1.0])
  lr_policy = random.choice([0.0002, 0.0002, 0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.002, 0.01])
  momentum = random.choice([0.8, 0.9])
  lr_baseline = random.choice([1.0])
  dropout_prob = random.choice([0.3])
  lr_lm = random.choice([0.1])
  batchSize = random.choice([1])
  command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", script, language, "L", entropy_weight, lr_policy, momentum, lr_baseline, dropout_prob, lr_lm, batchSize, model, BASE_DIR])
  subprocess.call(command)


 
