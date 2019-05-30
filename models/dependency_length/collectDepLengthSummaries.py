import os

path = "/u/scr/mhahn/deps/dependency_length/"

files = os.listdir(path+"/summaries/")
files = filter(lambda x:"SHORT" in x and "ByType" in x, files)

cache = {}

outHeader = ["Language", "FileName","ModelName","Counter","Model", "Temperature","OriginalCounter","AverageLengthPerWord", "AverageLengthPerSentence", "OriginalLoss"]

from numpy import median


with open(path+"depLength-auto-summary-full.tsv", "w") as outFile:
  print >> outFile, "\t".join(outHeader) #, "FileName", "ModelName", "Counter", "AverageLoss", "Head", "DH_Weight", "Dependency", "Dependent", "DistanceWeight"])
  for filename in files:
     if "SHORT" in filename:
        print filename
        with open(path+"/summaries/"+filename, "r") as inFile:
            try:
              header = next(inFile).strip().split("\t")
              text = next(inFile).strip().split("\t")
            except StopIteration:
              print ["EMPTY FILE?",path+filename]
              continue
            if len(header) < len(outHeader):
               header.append("OriginalLoss")
               text.append("NA")
            assert tuple(header) == tuple(outHeader), (header, outHeader)
            assert len(text) == len(outHeader), (text, outHeader)
            print >> outFile, "\t".join(map(str,text))




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

