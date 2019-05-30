import sys

language = sys.argv[1]

import subprocess

# RANDOM
modelDir = "manual_output_funchead_RANDOM"
with open("/u/scr/mhahn/deps/plane-parse.tsv", "r") as inFile:
   models = [x.split("\t") for x in inFile.read().strip().split("\n")]

   header = dict(zip(models[0], range(len(models[0]))))
   models = [x[header["Model"]] for x in models[1:] if x[header["Type"]] == modelDir and x[header["Language"]] == language]

import os
for modelID in models:
    files = os.listdir("/u/scr/mhahn/cky/")
    if len([x for x in files if "_"+modelID+"_" in x]) > 0:
      continue
    subprocess.call(["./python27", "chart_binarized.py", language, modelID, modelDir])










