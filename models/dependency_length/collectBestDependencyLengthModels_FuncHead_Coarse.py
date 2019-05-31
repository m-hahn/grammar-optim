# TODO what does this file do

import os

dirs = []
dirs.append("manual_output_funchead_two_coarse_final")
dirs.append("manual_output_funchead_two_coarse_parser_final")
dirs.append("manual_output_funchead_langmod_coarse_final")
dirs.append("REAL_REAL")
dirs.append("manual_output_funchead_langmod_coarse_tuning")
dirs.append("manual_output_funchead_RANDOM")
dirs.append("manual_output_funchead_coarse_depl")
dirs.append("manual_output_funchead_ground_coarse_final")

dirs.append("manual_output_funchead_langmod_coarse_best_balanced")
dirs.append("manual_output_funchead_two_coarse_parser_best_balanced")
dirs.append("manual_output_funchead_two_coarse_lambda09_best_balanced")



# find relevant model IDs

summariesDir = "../../grammars/dependency_length/summaries/"
summaries = os.listdir(summariesDir)

outPath = "../../grammars/dependency_length/total_summary_funchead_coarse.tsv"
with open(outPath, "w") as outFile:
 globalHeader = "\t".join(map(str,["Language", "FileName","ModelName","Counter", "Model", "Temperature", "OriginalCounter", "AverageLengthPerWord", "AverageLengthPerSentence", "OriginalLoss"]))
 print >> outFile, globalHeader+"\tType"
 for BASE_DIR in dirs:
    import os
    
    import subprocess
    import random
    
    from math import exp, sqrt
    import sys
    
    modelNumbers = None
    if BASE_DIR in ["REAL_REAL", "RANDOM"]:
       modelsProcessed = []
       languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Old_Church_Slavonic", "Ancient_Greek"] 
       assert len(languages) == 51
       


       for language in languages:
          modelsProcessed.append((language, BASE_DIR))
       models = modelsProcessed
    else:   
       languages = None
       
       models =[x for x in  os.listdir("../../raw-results/"+BASE_DIR+"/") if x.endswith(".tsv")]
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
             if len(parts) < 4 or parts[1] != "RANDOM" or parts[2] != "model":
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
    print(models) 

    for language, model in models:
       relevantResults = [x for x in summaries if x.startswith(language+"_") and x.endswith("_"+model+"_SHORT.tsv")]
       if len(relevantResults) == 0:
         continue
       fileName = relevantResults[0]
       with open(summariesDir+"/"+fileName, "r") as inFile:
          header = next(inFile).strip()
          assert header == globalHeader, (header, globalHeader)
          result = next(inFile).strip()
          print >>outFile, (result+"\t"+BASE_DIR)

print(outPath)
