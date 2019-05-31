import os

BASE_DIR = "manual_output_funchead_coarse_depl"
inPath = "../../../raw-results/"+BASE_DIR
outBasePath = "../../../grammars/"+BASE_DIR

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


inHeader = ["FileName", "ModelName", "Counter", "DH_Weight", "Dependency", "Dependent", "DistanceWeight"]
outHeader = ["Language"] + inHeader


outpath = outBasePath+"auto-summary-lstm.tsv"
with open(outpath, "w") as outFile:
  print >> outFile, "\t".join(outHeader) #, "FileName", "ModelName", "Counter", "AverageLoss", "Head", "DH_Weight", "Dependency", "Dependent", "DistanceWeight"])
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
#            assert missingColumns >= 0, [inHeader, header, set(header) - set(inHeader)]
            
            for i in range(len(header)):
              if header[i] in ["AverageLength", "Perplexity"]:
                  header[i] = "AverageLoss"
            if len(set(header) - set(inHeader)) > 0:
              print set(header) - set(inHeader)
#            matched = zip(inHeader+([None]*missingColumns), header)          
 #           print filter(lambda x:x[0] != x[1], matched)
            lineCounter = 0
            if "Pukwac" in filename:
               language = "Pukwac" 

            for line in inFile:
               lineCounter += 1
               line = line.strip().split("\t")
               outLine = [language] #, extractModelType(line[1])]
               for colName in inHeader:
                  if colName == "Language" and "Pukwac" in filename:
                      outLine.append("Pukwac")
                      continue
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
#                    print colName
#               outLine += (["NA"]*missingColumns)
               assert len(outLine) == len(outHeader)
               print >> outFile, "\t".join(outLine)



#> unique(data$ModelName)
# [1] 
# [2] 
# [3] 
# [4] 
# [5] 
# [6] 
# [7] 
# [8] 
# [9] 
#[10] 
#[11] 
#[12] 
#[13] 
#[14] 
#[15] 
#15 Levels: readDataDistEnglishGPUFree.py ...
print(outpath)
