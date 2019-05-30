#echo -e "Language\tModel\tType\tUAS\tPars" > ~/scr/deps/plane-parse.tsv
#./python27 assemblePlane_parse.py manual_output_funchead_two_coarse_final >> ~/scr/deps/plane-parse.tsv
#./python27 assemblePlane_parse.py manual_output_funchead_two_coarse_parser_final >> ~/scr/deps/plane-parse.tsv
#./python27 assemblePlane_parse.py manual_output_funchead_langmod_coarse_final >> ~/scr/deps/plane-parse.tsv
#./python27 assemblePlane_parse.py REAL_REAL >> ~/scr/deps/plane-parse.tsv
#./python27 assemblePlane_parse.py manual_output_funchead_langmod_coarse_tuning >> ~/scr/deps/plane-parse.tsv




import os
import sys

#print("manual_output_funchead_two_coarse_final")
#print("manual_output_funchead_two_coarse_parser_final")


dirs = []
dirs.append("manual_output_funchead_RANDOM2")
#dirs.append("manual_output_funchead_two_coarse_parser_best_balanced")
#dirs.append("manual_output_funchead_langmod_coarse_best_balanced")

print("/u/scr/mhahn/deps/plane-parse-random2.tsv")
with open("/u/scr/mhahn/deps/plane-parse-random2.tsv", "w") as outFile:
  print >> outFile, "Language\tModel\tType\tUAS\tPars"
  for BASE_DIR in dirs:
    
  
   languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
   assert len(languages) == 51, len(languages)
   
   import os
   
   import subprocess
   import random
   
   from math import exp, sqrt
   import sys
   found = 0 
   modelNumbers = None
   if BASE_DIR in ["REAL_REAL", "RANDOM"]:
      modelsProcessed = []
      for language in languages:
         modelsProcessed.append((language, BASE_DIR))
      models = modelsProcessed
   else:   
      languages = None
      
      models =[x for x in  os.listdir("/u/scr/mhahn/deps/"+BASE_DIR+"/") if x.endswith(".tsv")]
      modelsProcessed = []
      for i in range(len(models)):
         if "ground" in BASE_DIR:
              if "_inferWeights" not in models[i]:
                 continue
              ind = models[i].index("_infer")
              language = models[i][:ind]
              number = models[i][:-4].split("_")[-1]

         elif "RANDOM" in BASE_DIR:
             parts = models[i].split("_")
         #    print(parts)
             if len(parts) < 4 or parts[-3] not in ["RANDOM", "RANDOM2"] or parts[-2] != "model":
                continue
             if parts[-1].endswith(".tsv"):
                number = parts[-1][:-4]
                language = "_".join(parts[:-3])
             else:
                continue
         else:

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
   
   
   planePath = "/u/scr/mhahn/deps/parsing-upos/"
   
   if True:
     existingFiles = os.listdir(planePath)
     done = 0
     for language, model in models:
       existing = [x for x in existingFiles if x.startswith("performance-"+language+"_soso") and "_"+model+".txt" in x]
       if len(existing) > 0:
          done += 1
   #       print(language,model)
          for name in existing:
             with open(planePath+name, "r") as inFile:
   
                 data = [x.split(" ") for x in inFile.read().strip().split("\n")]
   #              print(data)
                 data = data[2:]
                 if len(data[0]) == 1:
                    continue
                 if float(data[0][-1]) < float(data[0][-2]):
                    continue
                 found+=1
                 print >> outFile, ("\t".join([language, model, BASE_DIR, data[1][-2], data[0][-2]]))
   print([BASE_DIR, found])
   #print("Done", done, "out of", 51*8)
