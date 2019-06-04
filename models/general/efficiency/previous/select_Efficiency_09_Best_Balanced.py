import os
import sys


languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Hebrew", "Hungarian", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu" , "Coptic", "Gothic",  "Latin", "Ancient_Greek", "Old_Church_Slavonic"]

import random
import subprocess

relevantPathFrom = "../../../raw-results/manual_output_funchead_two_coarse_lambda09_best/"
relevantPathTo = "../../../raw-results/manual_output_funchead_two_coarse_lambda09_best_balanced/"

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
           if line[8] == "obj":
             dhWeight = float(line[7])
             if dhWeight < 0:
                direction = 1
             elif dhWeight > 0:
                direction = -1
             break
       if direction is not None:
          if count[direction] < 4:
              copyfile(relevantPathFrom+name, relevantPathTo+name)
              count[direction] += 1
   assert count[1] == 4
   assert count[-1] == 4

  
