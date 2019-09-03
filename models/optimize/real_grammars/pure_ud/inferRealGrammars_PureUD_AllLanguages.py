languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]

assert len(languages) == 51
import os
import random

files = os.listdir("../../../../raw-results/manual_output_funchead_ground_coarse_pureUD/")

import subprocess

failures = 0
while failures < 1000:
  language = random.choice(languages)
  relevant = [x for x in files if x.startswith(language+"_infer")]
  if len(relevant) > 0:
     failures += 1
     continue
  subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "inferRealGrammars_PureUD.py", language, language])
#  break

