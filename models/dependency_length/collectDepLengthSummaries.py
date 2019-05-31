import os

pathIn = "../../raw-results/dependency_length/"
pathOut = "../../grammars/dependency_length/"

files = os.listdir(pathIn+"/summaries/")
files = filter(lambda x:"SHORT" in x and "ByType" in x, files)

cache = {}

outHeader = ["Language", "FileName","ModelName","Counter","Model", "Temperature","OriginalCounter","AverageLengthPerWord", "AverageLengthPerSentence", "OriginalLoss"]

from numpy import median


with open(pathOut+"depLength-auto-summary-full.tsv", "w") as outFile:
  print >> outFile, "\t".join(outHeader) 
  for filename in files:
     if "SHORT" in filename:
        print filename
        with open(pathIn+"/summaries/"+filename, "r") as inFile:
            try:
              header = next(inFile).strip().split("\t")
              text = next(inFile).strip().split("\t")
            except StopIteration:
              print ["EMPTY FILE?",pathIn+filename]
              continue
            if len(header) < len(outHeader):
               header.append("OriginalLoss")
               text.append("NA")
            assert tuple(header) == tuple(outHeader), (header, outHeader)
            assert len(text) == len(outHeader), (text, outHeader)
            print >> outFile, "\t".join(map(str,text))




