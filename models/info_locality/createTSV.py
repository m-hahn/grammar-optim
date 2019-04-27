
import os

inpath = "/u/scr/mhahn/deps/info_locality/"
outpath = "../results/info-locality/summary.tsv"
files = sorted(os.listdir(inpath))

with open(outpath, "w") as outFile:
  print >> outFile, "\t".join(["Language", "Model", "Type", "BigramCE"])
  for name in files:
    if "BigramCE" in name:
        with open(inpath+name, "r") as inFile:
           print >> outFile, next(inFile).strip()



