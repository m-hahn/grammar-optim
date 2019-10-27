import os
import sys





BASE_DIR = sys.argv[1]

# This is for cases where only the per-language optimal models require calculation of predictability.

if len(sys.argv) > 2:
   listPath = sys.argv[2] # e.g. writeup/best-parse-best-balanced.csv
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





if BASE_DIR == "REAL_REAL":
   assert False
   languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
   assert len(languages) == 51, len(languages)

   models = [(language, "REAL_REAL") for language in languages]
else:
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
          parts = models[i].split("_")
    #      print parts 
          if len(parts) < 4 or parts[-3] not in ["RANDOM", "RANDOM2", "RANDOM3", "RANDOM4", "RANDOM5"] or parts[-2] != "model":
             continue
          if parts[-1].endswith(".tsv"):
             number = parts[-1][:-4]
             language = "_".join(parts[:-3])
#             if "Ancient" in models[i]:
#                print(models[i])
#                print(language)
#                quit()

          else:
             continue
   #       print(number, language)
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

import os

script = "estimateParseability_ByBranchingEntropy.py"



failures = 0

if listPath is None:
   relevantModels = models

resbasepath = "../../../../raw-results/parsing-nondeterministic/"
while failures < 30000:
  existingFiles = os.listdir(resbasepath)
  language, model = random.choice(relevantModels)
  if languages is not None and language not in languages:
      assert False
      continue
  if language not in ["English", "Japanese", "Czech"]:
     failures += 1
     continue
  existing = [x for x in existingFiles if x.startswith("performance-"+language+"_") and "_"+model+".txt" in x]
  existing = [x for x in existing if len(open(resbasepath+x, "r").read().strip().split("\n")) >= 7]
  print(resbasepath, existing)
  if len(existing) > 0: #random.random() > ((1.0/(1+len(existing)))):
     print("Language model for this model exists "+str(((1.0/(1+len(existing))))))
     failures += 1
     continue
  failures = 0
  lr_policy = random.choice([0.002, 0.001, 0.001, 0.0005, 0.0005, 0.0005]) #random.choice([0.01, 0.001])
  entropy_weight = random.choice([1.0, 0.1, 0.01, 0.001, 0.001, 0.001, 0.0001])
  
  parameters = map(str, [lr_policy,	0.9,	entropy_weight,	0.001,	0.9,	0.999,	0.2,	20,	15,	2,	100,	2,	True,	200,	300,	0.0,	300])
  
  max_updates = 200000000
  

  command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", script, language, "NONE"] + parameters + [max_updates, model, BASE_DIR]) 
  print " ".join(command)
  subprocess.call(command)
 
