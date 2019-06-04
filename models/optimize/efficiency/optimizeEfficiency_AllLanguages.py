import os
import sys


languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Hebrew", "Hungarian", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu" , "Coptic", "Gothic",  "Latin", "Ancient_Greek", "Old_Church_Slavonic"]

import random
import subprocess
import os

with open("../../../results/strongest_models/best-two.csv", "r") as inFile:
  best = [x.replace('"', "").split(",") for x in inFile.read().strip().split("\n")]
  header_best = best[0]
  best = best[1:]
  bestByLanguage = {language : sorted([x[3] for x in best if x[1] == language]) for language in languages}


with open("../../chosen_hyperparameters/commands-lambda1.csv", "r") as inFile:
  config = [x.replace('"', "").split(",") for x in inFile.read().strip().split("\n")]
  header_config = config[0]
  config = config[1:]
  byLanguage = {language : sorted([tuple(x) for x in config if x[2] == language and x[1] in bestByLanguage[language]]) for language in languages}

#print(byLanguage)
#quit()
relevantPath = "../../../raw-results/manual_output_funchead_two_coarse_lambda09_best_large/"

while len(languages) > 0:
   script = "optimizeEfficiency.py"

   language = random.choice(languages)
   import os
   files = [x for x in os.listdir(relevantPath) if x.startswith(language+"_")]
   posCount = 0
   negCount = 0
   for name in files:
     with open(relevantPath+name, "r") as inFile:
       for line in inFile:
           line = line.split("\t")
           if line[8] == "obj":
             dhWeight = float(line[7])
             if dhWeight < 0:
                negCount += 1
             elif dhWeight > 0:
                posCount += 1
             break
   
   print([language, "Neg count", negCount, "Pos count", posCount])
   if negCount >= 12 and posCount >= 12:
       languages.remove(language) 
       continue

   args = {}

   # = random.choice(languages)
   line = byLanguage[language]

   if len(line) > 0:
      command = random.choice(line)[-1]
   else:
      continue
   print((" "+(command.strip())).split(" --") )
   command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", script] + command.split(" ")  )

   print(command)
   print(" ".join(command))
   subprocess.call(command)
#   break 
