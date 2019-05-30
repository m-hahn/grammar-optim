import os

path = "/u/scr/mhahn/deps/dependency_length/"

files = os.listdir(path)
files = filter(lambda x:"ByType" in x, files)
files = filter(lambda x:"SHORT" not in x, files)

cache = {}

inHeader = ["FileName","ModelName","Counter","Model","SentenceNumber","Type","Length", "Temperature", "OriginalCounter"]

outHeader = ["Language", "FileName","ModelName","Counter","Model", "Temperature","OriginalCounter", "AverageLengthPerWord", "AverageLengthPerSentence"]


with open(path+"depLength-auto-summary.tsv", "w") as outFile:
  print >> outFile, "\t".join(outHeader) #, "FileName", "ModelName", "Counter", "AverageLoss", "Head", "DH_Weight", "Dependency", "Dependent", "DistanceWeight"])
  for filename in files:
     if "ONLY" in filename:
        continue
     if "model" in filename:
        print filename
        part1 = filename.split("_model_")[0]
#        if "_" in part1:
        language = part1.split("_")[0]
#        else:
#          language = "English"
        with open(path+filename, "r") as inFile:
            try:
              header = next(inFile).strip().split("\t")
            except StopIteration:
              print ["EMPTY FILE?",path+filename]
              continue
            #print header
            missingColumns = len(inHeader) - len(header)
            assert missingColumns >= 0, [inHeader, header, set(header) - set(inHeader)]
            
            if len(set(header) - set(inHeader)) > 0:
              print set(header) - set(inHeader)
            lineCounter = 0
            fileName = None
            modelName = None
            counter = None
            model = None
#            lengthByType = {}
#            lengthByLength = {}
#            lastSentenceNumber = None
#            lengthOfSentence = None
            totalDependencyLength = 0
            totalNumberOfWords = 0
            perType = {}
            with open(path+"/summaries/"+filename.replace(".tsv", "_SHORT.tsv"), "w") as outFile2:
               print >> outFile2, "\t".join(outHeader) #, "FileName", "ModelName", "Counter", "AverageLoss", "Head", "DH_Weight", "Dependency", "Dependent", "DistanceWeight"])
             

               for line in inFile:
                  line = line.strip().split("\t")
                  totalDependencyLength += int(line[header.index("Length")])
                  totalNumberOfWords += 1
                  lineCounter += 1
                                 
                  newSentenceNumber = line[header.index("SentenceNumber")]
               outLine = [language] #, extractModelType(line[1])]
               for colName in inHeader:
                  if colName == "Model" and colName not in header:
                    outLine.append("REAL")
                  elif colName == "Temperature" and colName not in header:
                    outLine.append(1.0)
                  elif colName == "OriginalCounter"  and colName not in header:
                    outLine.append("NA")
                  else:
                     if colName in outHeader:
                          i = inHeader.index(colName)
                          outLine.append(line[i])
               if totalNumberOfWords == 0:
                   print "Apparently empty file"
                   continue
               outLine.append(float(totalDependencyLength)/totalNumberOfWords)
               outLine.append(float(totalDependencyLength)/int(newSentenceNumber))
   
   #outH   eade ["Language", "FileName","ModelName","Counter","Model","AverageLengthPerWord", "AverageLengthPerSentence", "Temperature"]
   
   
               assert len(outLine) == len(outHeader)
               print >> outFile, "\t".join(map(str,outLine))
               print >> outFile2, "\t".join(map(str,outLine))





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

