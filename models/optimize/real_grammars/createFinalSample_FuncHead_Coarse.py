languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]

assert len(languages) == 51
import os
import random

inDir = "../../raw-results/manual_output_funchead_ground_coarse/"
files = os.listdir(inDir)

outDir = "../../raw-results/manual_output_funchead_ground_coarse_final/"

import subprocess

import shutil

for language in languages:
   relevant = [x for x in files if x.startswith(language+"_infer")]
   chosen = random.choice(relevant)
   with open(inDir+chosen, "r") as inFile:
      next(inFile)
      counter = int(next(inFile).split("\t")[2])
   print(counter)
   assert counter > 200000
   shutil.copyfile(inDir+chosen, outDir+chosen)



