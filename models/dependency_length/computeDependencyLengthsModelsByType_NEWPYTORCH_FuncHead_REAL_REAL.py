# Compute dependency length for the actual orderings found in the 51 corpora

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



inpModels_path = "../../raw-results/"+BASE_DIR
for language in languages[int(starting*len(languages)):int(ending*len(languages))]:
  relevantRuns = [x for x in os.listdir("../../grammars/dependency_length/summaries/") if (language in x) and ("REAL_REAL" in x)]
  if len(relevantRuns) > 0:
      print("Skipping model")
      continue
  print "Doing model"
  print language 
  assert temperature == "Infinity"
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "computeDependencyLengths/computeDependencyLengthsByType_NEWPYTORCH_Deterministic_FuncHead_REAL_REAL.py", language, language, "REAL_REAL", str(temperature), "REAL_REAL"])



