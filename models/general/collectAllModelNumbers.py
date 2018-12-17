import os

print "Skipping filenames containing 'Morphology', 'FastForward'"

path = "/juicier/scr120/scr/mhahn/deps/"
files = filter(lambda x:x.startswith("LOG"), os.listdir(path))
counter = 0
with open(path+"/models_log_summary.tsv", "w") as outFile:
  print >> outFile, "\t".join(["FileName", "Model", "Language", "Command"])
  for fileName in files:
     counter += 1
     if counter % 1000 == 0:
         print float(counter)/len(files)
     if "Morphology" in fileName or 'FastForward' in fileName:
          continue
     with open(path+fileName, "r") as inFile:
         command = inFile.read().strip()
  #       print (fileName, command)
         parts = fileName[3:-4].split("_")
  #       language = parts[0]
         myID = parts[-1]
         command = command.split(" ")
         modelName = command[0]
  #       if language != command[1]:
  #           print (language, command[1])
         if len(command) < 2:
            print("error", fileName)
            continue
         language = command[1]
         #print command[1:]
         #for i in range(2,len(command)):
         #  if command[i][0] == command[i][0].upper():
         #      language += "_"+command[i]
         #  else:
         #      break
         #if i > 2:
         #    print language
         command = " ".join(command[1:])
         print >> outFile, "\t".join([myID, modelName, language, command])

# ('LOGLithuanian_readDataDistCrossLGPUDepLengthMomentumEntropyUnbiasedBaseline_OrderBugFixed_NoPunct_NEWPYTORCH_AllCorpPerLang_BoundIterations.py_model_9561466.txt', 'readDataDistCrossLGPUDepLengthMomentumEntropyUnbiasedBaseline_OrderBugFixed_NoPunct_NEWPYTORCH_AllCorpPerLang_BoundIterations.py Lithuanian lt 0.001 0.1 0.9')

