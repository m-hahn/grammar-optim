


import os
import sys

#print("manual_output_funchead_two_coarse_final")
#print("manual_output_funchead_two_coarse_parser_final")

BASE_DIR = sys.argv[1]

languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
assert len(languages) == 51, len(languages)
#else:
#   languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Kazakh", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic"]
#
#if len(sys.argv) > 2:
#  modelNumbers = sys.argv[2].split(",")
#else:
#  modelNumbers = None
#

import os

import subprocess
import random

from math import exp, sqrt
import sys

modelNumbers = None
if BASE_DIR == "REAL_REAL":
   modelsProcessed = []
   for language in languages:
      modelsProcessed.append((language, "REAL_REAL"))
   models = modelsProcessed
else:   
   languages = None
   
   models =[x for x in  os.listdir("../../raw-results/"+BASE_DIR+"/") if x.endswith(".tsv")]
   modelsProcessed = []
   for i in range(len(models)):
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
models = [model for model in models if len(model) == 2]

import os
parsingDone = [x.split("_")[-1][:-4] for x in os.listdir("../../raw-results/language_modeling_coarse_plane/") if "ZeroTemp" in x]


scripts = ["readDataDistCrossGPUFreeMomentumEarlyStopEntropyPersevereAnneal_OrderBugfix_Fixed_NoPunct_AllCorpPerLang_NEWPYTORCH_Corrected_FastAsBefore_Zero_Running_FuncHead_LANGMOD_ZeroTemp_COARSE_PLANE.py"]



#print(parsingDone)

planePath = "../../raw-results/language_modeling_coarse_plane/"

if True:
  existingFiles = os.listdir(planePath)
  script = random.choice(scripts) #scripts[0] if random.random() < 0.8 else scripts[1]
  done = 0
  for language, model in models:
    existing = [x for x in existingFiles if x.startswith(language) and "_"+model+"_" in x]
 #   if len(existing) > 1:
  #     print(len(existing))
    if len(existing) > 0:
       done += 1
#       print(language,model)
       for name in existing:
          with open(planePath+name, "r") as inFile:

              data = [x.split("\t") for x in inFile.read().strip().split("\n")]
#              print(data)
              data = data[1:]
              if len(data[0]) == 1:
                 continue
              if float(data[0][-1]) < float(data[0][-2]):
                 continue
              print("\t".join([language, model, BASE_DIR, data[1][-2]]))
#print("Done", done, "out of", 51*8)
