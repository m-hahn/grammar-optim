import subprocess
import os
import sys

if sys.argv[1] == "Infinity":
   temperature = sys.argv[1]
else:
   temperature = int(sys.argv[1])
starting = float(sys.argv[2])
assert starting >= 0
assert starting <= 1
ending = float(sys.argv[3])
assert ending >= 0
assert ending <= 1

BASE_DIR = sys.argv[4]

#skipOldOnes = (sys.argv[2] == "True")
#
#if skipOldOnes:
#   outputsCreatedSoFar = filter(lambda x:"SHORT" in x and "NEWPYTORCH" in x, os.listdir("/juicier/scr120/scr/mhahn/deps/"+"/dependency_length/summaries/"))
#   for i in range(len(outputsCreatedSoFar)):
#      print "/juicier/scr120/scr/mhahn/deps/"+"/dependency_length/summaries/"+outputsCreatedSoFar[i]
#      try:
#        with open("/juicier/scr120/scr/mhahn/deps/"+"/dependency_length/summaries/"+outputsCreatedSoFar[i], "r") as inFile:
#           header = next(inFile).strip().split("\t")
#           line = next(inFile).strip().split("\t")
#           temperatureHere = line[header.index("Temperature")]
#        parts = outputsCreatedSoFar[i].replace(".tsv", "").split("_")
#        assert parts[2] == "NEWPYTORCH.py"
#        assert parts[3] == "model"
#        outputsCreatedSoFar[i] = (parts[0], parts[5], int(float(temperatureHere)) if temperatureHere != "Infinity" else "Infinity")
#      except StopIteration:
#        continue
#   outputsCreatedSoFar = set(outputsCreatedSoFar)
#   print outputsCreatedSoFar

inpModels_path = "/juicier/scr120/scr/mhahn/deps/"+BASE_DIR
models = os.listdir(inpModels_path)
print("Files in models dir", len(models))
if "manual_output_funchead_RANDOM" == BASE_DIR:
  models = filter(lambda x:"RANDOM_model_" in x, models)
elif "manual_output_funchead_ground_coarse_final" == BASE_DIR:
  models = filter(lambda x:"_inferWei" in x, models)
else:
  models = filter(lambda x: "readData" in x, models) # "NoPunct" in x and 
models = sorted(models)
print models
print(len(models))
for model in models[int(starting*len(models)):int(ending*len(models))]:
  model = model.replace(".tsv", "")
  if "manual_output_funchead_RANDOM" == BASE_DIR:
    language = model[:model.index("_RANDOM")]
  elif "manual_output_funchead_ground_coarse_final" == BASE_DIR:
    language = model[:model.index("_inferWe")]
  else:
    language = model[:model.index("_read")]
  model = model.split("_")
#  language = model[0]
  modelNumber = model[-1]
  relevantRuns = [x for x in os.listdir("/u/scr/mhahn/deps/dependency_length/summaries/") if "_"+modelNumber+"_" in x]
  if len(relevantRuns) > 0:
      print("Skipping model")
      continue
  #relevantRuns = [x for x in os.listdir("/u/scr/mhahn/deps/dependency_length") if x "_"+modelNumber+".tsv"]
  #if len(relevantRuns) > 0:
  #    continue
#  if skipOldOnes:
#    if (language, modelNumber, temperature) in outputsCreatedSoFar:
#      print "Skipping model "+str(skipOldOnes)
#      continue
  print "Doing model"
  print model
#  continue
  assert temperature == "Infinity"
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "computeDependencyLengths/computeDependencyLengthsByType_NEWPYTORCH_Deterministic_FuncHead_Coarse.py", language, language, modelNumber, str(temperature), BASE_DIR])


#Czech_computeDependencyLengthsByType_NEWPYTORCH.py_model_3931200_2937602.tsv

