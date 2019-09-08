languages = set(["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"])

assert len(languages) == 51



languages = list(languages)

assert len(languages) == len(set(languages))

import os
import random


import subprocess

modelsDir = "../../../../raw-results/manual_output_funchead_ground_coarse_pureUD/"
modelsDirOut = "../../../../raw-results/manual_output_funchead_ground_coarse_pureUD_final/"

files = os.listdir(modelsDir)
import shutil

for language in languages:
  relevant = [x for x in files if x.startswith(language+"_infer")]
  assert len(relevant) > 0, language
  relevantModelExists = False
  farthestName, farthestCounter = None, 0
  for filename in relevant:
      with open(modelsDir+filename, "r") as inFile:
         header = next(inFile).strip().split("\t")
         line = next(inFile).strip().split("\t")
         counter = int(line[header.index("Counter")])
         print(counter)
         if counter > farthestCounter:
           farthestName = filename
           farthestCounter = counter

  print(farthestName, farthestCounter)
  shutil.copyfile(modelsDir+farthestName, modelsDirOut+farthestName)
