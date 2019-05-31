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



inpModels_path = "../../raw-results/"+BASE_DIR
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
  modelNumber = model[-1]
  relevantRuns = [x for x in os.listdir("../../raw-results/dependency_length/summaries/") if "_"+modelNumber+"_" in x]
  if len(relevantRuns) > 0:
      print("Skipping model")
      continue
  print "Doing model"
  print model
  assert temperature == "Infinity"
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "computeDependencyLengths/computeDependencyLengthsByType_NEWPYTORCH_Deterministic_FuncHead_Coarse.py", language, language, modelNumber, str(temperature), BASE_DIR])


