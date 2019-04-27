

import subprocess
import os
import sys

temperature = "Infinity"


languages = ["English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Old_Church_Slavonic", "Ancient_Greek", "Hindi", "Swedish", "German", "Urdu"]
assert len(languages) == 51

for language in languages:
   #language = sys.argv[1]
   
   
   
   
   models = []
   
   
   
   
   
   # random
   #
   # optimized
   objectives = ["best-"+x+".csv" for x in ["depl", "langmod-best-balanced", "parse-best-balanced", "two_coarse_lambda09_best_large"]] + ["models-mle.csv"]
   for obj in objectives:
      with open("../results/strongest_models/"+obj, "r") as inFile:
         data = [x.replace('"',"").split(",") for x in inFile.read().strip().split("\n")]
      header =  dict(zip(data[0], range(len(data[0]))))
      for line in data[1:]:
       if line[header["Language"]] == language:
          models.append((line[header["Model"]], line[header["Type"]]))

   modelsRand = set()
   for name in [""]: #, "-random2"]:
     with open("/u/scr/mhahn/deps/plane-fixed"+name+".tsv", "r") as inFile:
        plane = [x.split("\t") for x in inFile.read().strip().split("\n")]
     header = dict(zip(plane[0], range(len(plane[0]))))
     for line in plane[1:]:
        if line[header["Language"]] == language and line[header["Type"]].startswith("manual_output_funchead_RANDOM"):
            modelsRand.add((line[header["Model"]], line[header["Type"]]))
   models = models + list(modelsRand)
   print(models)   
   #print(models)
   #outpath = "../../results/info-locality/"+language+".tsv"
   #print(outpath)
   #with open(outpath, "w") as outFile:
   #  print >> outFile, "\t".join(["Model", "Type"])
   #  for model, typ in models:
   #    print >> outFile, "\t".join([model, typ])
   
   # real language
   
   for modelNumber, BASE_DIR in models:
     assert temperature == "Infinity"
     subprocess.call(["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "info_locality/computeBigramCE.py", language, language, modelNumber, str(temperature), BASE_DIR])
   
   
   #Czech_computeDependencyLengthsByType_NEWPYTORCH.py_model_3931200_2937602.tsv
   
