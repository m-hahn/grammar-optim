
assert False, "No need to run this script again."


import subprocess
import os
import sys

temperature = "Infinity"
language = sys.argv[1]


models = []

# random
with open("../../grammars/plane/plane-fixed.tsv", "r") as inFile:
   plane = [x.split("\t") for x in inFile.read().strip().split("\n")]
header = dict(zip(plane[0], range(len(plane[0]))))
for line in plane[1:]:
   if line[header["Language"]] == language and line[header["Type"]] == "manual_output_funchead_RANDOM":
       models.append((line[header["Model"]], line[header["Type"]]))
#
# optimized
objectives = ["best-"+x+".csv" for x in ["depl", "langmod-best-balanced", "parse-best-balanced", "two-lambda09-best-balanced"]] + ["models-mle.csv"]
for obj in objectives:
   with open("writeup/"+obj, "r") as inFile:
      data = [x.replace('"',"").split(",") for x in inFile.read().strip().split("\n")]
   header =  dict(zip(data[0], range(len(data[0]))))
   for line in data[1:]:
    if line[header["Language"]] == language:
       models.append((line[header["Model"]], line[header["Type"]]))
    

#print(models)
outpath = "../results/dependency-length/forVisualization/"+language+"_forVisualization.tsv"
print(outpath)
with open(outpath, "w") as outFile:
  print >> outFile, "\t".join(["Model", "Type"])
  for model, typ in models:
    print >> outFile, "\t".join([model, typ])

# real language

for modelNumber, BASE_DIR in models:
  assert temperature == "Infinity"
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "computeDependencyLengths/computeDependencyLengths_ForGrammars.py", language, language, modelNumber, str(temperature), BASE_DIR, "True"])


#Czech_computeDependencyLengthsByType_NEWPYTORCH.py_model_3931200_2937602.tsv

