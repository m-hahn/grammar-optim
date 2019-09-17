import os
import sys

import subprocess
import random

from math import exp, sqrt

BASE_DIR = "REAL_REAL"

scripts = ["estimatePredictability_WithoutPOS.py"]

languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
assert len(languages) == 51, len(languages)


failures = 0
random.shuffle(languages)
basepath = "../../../../raw-results//language_modeling_coarse_plane_fixed_withoutPOS"
for language in languages:
  existingFiles = os.listdir(basepath)
  script = random.choice(scripts) #scripts[0] if random.random() < 0.8 else scripts[1]
  existing = [x for x in existingFiles if x.startswith(language)]
  exists = False
  for filename in existing:
     with open(basepath+"/"+filename, "r") as inFile:
         if "REAL_REAL" in inFile.read():
            exists = True
            break
  if exists:
     print("Language model for this model exists "+str(((1.0/(1+len(existing))))))
     failures += 1
     continue
  failures = 0
  entropy_weight = random.choice([0.001, 0.001,0.001, 0.001,  0.01, 0.1, 1.0])
  lr_policy = random.choice([0.0002, 0.0002, 0.0005, 0.0005, 0.001, 0.001, 0.001, 0.001, 0.002, 0.01])
  momentum = random.choice([0.8, 0.9])
  lr_baseline = random.choice([1.0])
  dropout_prob = random.choice([0.3])
  lr_lm = random.choice([0.1])
  batchSize = random.choice([1])
  command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", script, language, "L", entropy_weight, lr_policy, momentum, lr_baseline, dropout_prob, lr_lm, batchSize, "REAL_REAL", BASE_DIR])
  subprocess.call(command)


 
