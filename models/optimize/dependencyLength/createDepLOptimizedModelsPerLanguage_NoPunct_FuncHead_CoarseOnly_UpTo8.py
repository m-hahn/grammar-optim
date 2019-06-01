import os
import subprocess
import sys
keys = {}
import sys

if len(sys.argv)>1:
   languages = sys.argv[1].split(",")
else:
   languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Hebrew", "Hungarian", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu" , "Coptic", "Gothic",  "Latin", "Ancient_Greek", "Old_Church_Slavonic"]


import random
import subprocess

relevantPath = "../../../raw-results/manual_output_funchead_coarse_depl/"

while len(languages) > 0:
   script = 'readDataDistCrossLGPUDepLengthMomentumEntropyUnbiasedBaseline_OrderBugFixed_NoPunct_NEWPYTORCH_AllCorpPerLang_BoundIterations_FuncHead_CoarseOnly.py'

   language = random.choice(languages)
   import os
   files = [x for x in os.listdir(relevantPath) if x.startswith(language+"_")]
   posCount = 0
   negCount = 0
   for name in files:
     with open(relevantPath+name, "r") as inFile:
       for line in inFile:
           line = line.split("\t")
           if line[5] == "obj":
             dhWeight = float(line[4])
             if dhWeight < 0:
                negCount += 1
             elif dhWeight > 0:
                posCount += 1
             break
   
   print([language, "Neg count", negCount, "Pos count", posCount])
   if negCount >= 4 and posCount >= 4:
       languages.remove(language) 
       continue


   LR_POLICY = random.choice(["0.1", "0.1", "0.01"])
   subprocess.call(['/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7', script, language, "L", "0.001", LR_POLICY, "0.9"])


