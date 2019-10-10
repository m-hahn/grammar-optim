# Preregistration for these optimized grammars: http://aspredicted.org/blind.php?x=bg35x7


import os
import sys


dirs = []
dirs.append("manual_output_funchead_two_coarse_lambda09_best_large")
dirs.append("manual_output_funchead_two_coarse_parser_best_balanced")
dirs.append("manual_output_funchead_langmod_coarse_best_balanced")
dirs.append("manual_output_funchead_RANDOM")
dirs.append("manual_output_funchead_RANDOM2")
dirs.append("manual_output_funchead_RANDOM3")
dirs.append("manual_output_funchead_RANDOM4")
dirs.append("manual_output_funchead_RANDOM5")
#dirs.append("manual_output_funchead_coarse_depl")
dirs.append("manual_output_funchead_ground_coarse_final")
#dirs.append("manual_output_funchead_RLR")
dirs.append("REAL_REAL")

outPath = "../../../../grammars/plane/controls/plane-fixed-nondeterministic.tsv"
with open(outPath, "w") as outFile:
  print >> outFile, "Language\tModel\tType\tSurp"
  for BASE_DIR in dirs:
    found = 0
    
    languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
    assert len(languages) == 51, len(languages)
    
    import os
    
    import subprocess
    import random
    
    from math import exp, sqrt
    import sys
    
    modelNumbers = None
    if BASE_DIR in ["REAL_REAL"]:
       modelsProcessed = []
       for language in languages:
          modelsProcessed.append((language, BASE_DIR))
       models = modelsProcessed
    else:   
       languages = None

       models =[x for x in  os.listdir("../../../../raw-results/"+BASE_DIR+"/") if x.endswith(".tsv")]
     
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
             if len(parts) < 4 or parts[-3] not in ["RANDOM", "RANDOM2", "RANDOM3", "RANDOM4", "RANDOM5"] or parts[-2] != "model":
                print(models[i])
                continue
             if parts[-1].endswith(".tsv"):
                number = parts[-1][:-4]
                language = "_".join(parts[:-3])
             else:
                continue
         elif "RLR" in BASE_DIR:
             parts = models[i].split("_")
         #    print(parts)
             if len(parts) < 4 or parts[-3] != "RLR" or parts[-2] != "model":
                print(models[i])
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
    assert len(models) > 0, BASE_DIR
    models = [model for model in models if len(model) == 2]
    
    import os
    
    
    
    planePath = "../../../../raw-results/language_modeling_coarse_plane_fixed_nondeterministic/"
    
    if True:
      existingFiles = os.listdir(planePath)
      done = 0
      for language, model in models:
        existing = [x for x in existingFiles if x.startswith(language) and "_"+model+"_" in x]
        if "RANDOM" in BASE_DIR:
           existing += [x for x in existingFiles if x.startswith(language) and ("_"+model+"." if "RANDOM" in BASE_DIR else "_"+model+"_") in x]
        if "RLR" in BASE_DIR:
           existing += [x for x in existingFiles if x.startswith(language) and ("_"+model+"." if "RLR" in BASE_DIR else "_"+model+"_") in x]

     #   if len(existing) > 1:
      #     print(len(existing))
        if len(existing) > 0:
           done += 1
    #       print(language,model)
           for name in existing:
              with open(planePath+name, "r") as inFile:
    
                  data = [x.split("\t") for x in inFile.read().strip().split("\n")]
    #              print(data)
                  data = data[1:]
                  if len(data[0]) == 1:
                     print("Unfinished model", name, BASE_DIR)
                     continue
                  if float(data[0][-1]) < float(data[0][-2]):
                     print("Unfinished model", name, BASE_DIR)
                     continue
                  found+=1
                  print >> outFile, ("\t".join([language, model, BASE_DIR, data[1][-2]]))
    print([BASE_DIR, found])

print(outPath)
