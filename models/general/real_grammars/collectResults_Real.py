import os

BASE_DIR = "manual_output_funchead_ground_coarse_final"
inPath = "../../../raw-results/"+BASE_DIR+"/"
outPath = "../../../grammars/"+BASE_DIR+"/"

files = os.listdir(inPath)

cache = {}

def extractModelTypeCached(modelName, objName):
   if (modelName, objName) not in cache:
     cache[(modelName,objName)] = extractModelType(modelName,objName)
   return cache[(modelName,objName)]
   

def extractModelType(modelName, objName):
   if modelName == "readDataDistCrossGPUFreeAllTwoEqual.py":
      return "Two"
   print ["UNKNOWN TYPE", modelName]
   return modelName


inHeader = ["FileName", "ModelName", "Counter", "Dependency", 'Distance_Mean_NoPunct', 'Distance_Sigma_NoPunct', 'DH_Sigma_NoPunct', 'DH_Mean_NoPunct']
outHeader = ["Language"] + inHeader



with open(outPath+"auto-summary-lstm.tsv", "w") as outFile:
  print >> outFile, "\t".join(outHeader) 
  for filename in files:
     if "model" in filename:
        print "READING "+filename 
        part1 = filename.split("_model_")[0]
        if "_" in part1:
          language = part1.split("_")[0]
        else:
          language = "English"
        with open(inPath+filename, "r") as inFile:
            try:
              header = next(inFile).strip().split("\t")
            except StopIteration:
              print ["EMPTY FILE?",inPath+filename]
              continue
            missingColumns = len(inHeader) - len(header)
            
            for i in range(len(header)):
              if header[i] in ["AverageLength", "Perplexity"]:
                  header[i] = "AverageLoss"
            if len(set(header) - set(inHeader)) > 0:
              print set(header) - set(inHeader)
            lineCounter = 0
            for line in inFile:
               lineCounter += 1
               line = line.strip().split("\t")
               outLine = [language] #, extractModelType(line[1])]
               for colName in inHeader:
                  try:
                    i = header.index(colName)
                    outLine.append(line[i])
                    if outLine[-1].startswith("["):
                        outLine[-1] = outLine[-1].replace("[","").replace("]","")
                    if colName == "ObjectiveName":
                       if line[i] != extractModelTypeCached(line[1], line[i]) and lineCounter == 1:
                          print [line[i], "CHOOSING INSTEAD", extractModelTypeCached(line[1], line[i])]
                       outLine[-1] = extractModelTypeCached(line[1], line[i])
                  except ValueError:
                    if colName == "ObjectiveName":
                       outLine.append(extractModelTypeCached(line[1], "NONE"))
                    else:
                       outLine.append("NA")
               assert len(outLine) == len(outHeader)
               print >> outFile, "\t".join(outLine)


