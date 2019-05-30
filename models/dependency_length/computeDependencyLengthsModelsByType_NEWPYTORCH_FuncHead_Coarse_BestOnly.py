import subprocess
import os
import sys

temperature = "Infinity"
starting = float(sys.argv[1])
assert starting >= 0
assert starting <= 1
ending = float(sys.argv[2])
assert ending >= 0
assert ending <= 1


models = []
for name in ["best-"+x+".csv" for x in ["langmod-best-balanced", "parse-best-balanced", "two-lambda09-best-balanced"]]:
   with open("writeup/"+name, "r") as inFile:
      data = [x.replace('"', "").split(",") for x in inFile.read().strip().split("\n")]
      header = dict(zip(data[0], range(len(data[0]))))
      for line in data[1:]:
         models.append((line[header["Language"]], line[header["Type"]], line[header["Model"]]))


models = sorted(models)
print models
print(len(models))
for model in models[int(starting*len(models)):int(ending*len(models))]:
  modelNumber = model[2]
  relevantRuns = [x for x in os.listdir("/u/scr/mhahn/deps/dependency_length/summaries/") if "_"+modelNumber+"_" in x]
  if len(relevantRuns) > 0:
      print("Skipping model")
      continue
  print "Doing model"
  print model
  assert temperature == "Infinity"
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "computeDependencyLengths/computeDependencyLengthsByType_NEWPYTORCH_Deterministic_FuncHead_Coarse.py", model[0], model[0], modelNumber, str(temperature), model[1]])


#Czech_computeDependencyLengthsByType_NEWPYTORCH.py_model_3931200_2937602.tsv

