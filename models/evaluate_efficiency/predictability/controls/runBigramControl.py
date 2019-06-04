import sys

language = sys.argv[1]
typ = sys.argv[2]
assert typ in ["two", "RANDOM"]

import subprocess

# RANDOM
if typ == "RANDOM":
   modelDir = "manual_output_funchead_RANDOM"
elif typ == "two":
   modelDir = "manual_output_funchead_two_coarse_lambda09_best_balanced"


if typ == "two":
   with open("writeup/best-two-lambda09-best-balanced.csv", "r") as inFile:
      models = [x.replace('"', "").split(",") for x in inFile.read().strip().split("\n")]
   
      header = dict(zip(models[0], range(len(models[0]))))
      models = [x[header["Model"]] for x in models[1:] if x[header["Type"]] == modelDir and x[header["Language"]] == language]
elif typ == "RANDOM":
   with open("../../../../grammars/plane/plane-parse.tsv", "r") as inFile:
      models = [x.split("\t") for x in inFile.read().strip().split("\n")]
   
      header = dict(zip(models[0], range(len(models[0]))))
      models = [x[header["Model"]] for x in models[1:] if x[header["Type"]] == modelDir and x[header["Language"]] == language]

import os
for modelID in models:
    files = os.listdir("../../../../raw-results/language_modeling_adversarial/")
    if len([x for x in files if "_"+modelID+"_" in x and "_dotdotdot_COARSE_PLANE_ngram.py_" in x]) > 0:
      continue
    subprocess.call(["./python27", "estimatePredictability_Bigrams.py", language, modelID, modelDir, "none"])










