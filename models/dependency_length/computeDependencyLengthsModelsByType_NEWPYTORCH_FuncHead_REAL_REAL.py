import subprocess
import os
import sys


languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Old_Church_Slavonic", "Ancient_Greek"] 
assert len(languages) == 51

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
assert BASE_DIR == "REAL_REAL"

# ./python27 computeDependencyLengths/computeDependencyLengthsModelsByType_NEWPYTORCH_FuncHead_REAL_REAL.py Infinity 0.0 0.1 REAL_REAL





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
for language in languages[int(starting*len(languages)):int(ending*len(languages))]:
  relevantRuns = [x for x in os.listdir("/u/scr/mhahn/deps/dependency_length/summaries/") if (language in x) and ("REAL_REAL" in x)]
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
  print language 
#  continue
  assert temperature == "Infinity"
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "computeDependencyLengths/computeDependencyLengthsByType_NEWPYTORCH_Deterministic_FuncHead_REAL_REAL.py", language, language, "REAL_REAL", str(temperature), "REAL_REAL"])


#Czech_computeDependencyLengthsByType_NEWPYTORCH.py_model_3931200_2937602.tsv

