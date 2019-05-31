# /u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7 generateManyModels_AllTwo.py
import os
import sys


languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Hebrew", "Hungarian", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu" , "Coptic", "Gothic",  "Latin", "Ancient_Greek", "Old_Church_Slavonic"]

import random
import subprocess

relevantPathFrom = "../../../raw-results/manual_output_funchead_two_coarse_parser_best/"
relevantPathTo = "../../../raw-results/manual_output_funchead_two_coarse_parser_best_balanced/"

from shutil import copyfile
for language in languages:
   import os
   files = [x for x in os.listdir(relevantPathFrom) if x.startswith(language+"_")]
   count = {-1:0, 1:0}
   for name in files:
     with open(relevantPathFrom+name, "r") as inFile:
       direction = None
       for line in inFile:
           line = line.split("\t")
           if line[7] == "obj":
             dhWeight = float(line[6])
             if dhWeight < 0:
                direction = 1
             elif dhWeight > 0:
                direction = -1
             break
       if direction is not None:
          if count[direction] < 4:
              copyfile(relevantPathFrom+name, relevantPathTo+name)
              count[direction] += 1
   print(language, count)
   assert count[1] == 4
   assert count[-1] == 4

  
