# For baseline languages

import sys

language = sys.argv[1]

import subprocess

arguments = "NONE 0.001 0.9 0.001 0.001 0.9 0.999 0.2 20 15 2 100 2 True 200 300 0.0 300 20000000".split(" ")


# RANDOM
modelDir = "manual_output_funchead_RANDOM"
with open("../../../grammars/plane/plane-parse.tsv", "r") as inFile:
   models = [x.split("\t") for x in inFile.read().strip().split("\n")]

   header = dict(zip(models[0], range(len(models[0]))))
   models = [x[header["Model"]] for x in models[1:] if x[header["Type"]] == modelDir and x[header["Language"]] == language]

for modelID in models:
  for controlType in ["inwards", "interleave", "even_odd",  "pos"]:
    subprocess.call(["./python27", ("estimate_Parseability_Reordered.py" if controlType != "pos" else "estimate_Parseability_Reordered_POS.py"), language] + arguments + [modelID, modelDir, controlType])










