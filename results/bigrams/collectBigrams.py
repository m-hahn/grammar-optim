
import os, sys

path = "/home/user/CS_SCR/deps/language_modeling_adversarial/"

with open("bigrams_results.tsv", "w") as outFile:
  print >> outFile, "\t".join(["Script", "Language", "Model", "Directory", "foo", "Loss", "LossWords", "LossPOS"])
  for name in sorted(os.listdir(path)):
    if "ngram" in name:
        with open(path+name, "r") as inFile:
            data = inFile.read().strip().split("\n")
            data[0] = data[0].split("\t")
            data[1:] = map(float, data[1:])
            if data[-1] == 0:
                continue
            print data
            print >> outFile, "\t".join(data[0] + map(str, data[1:]))



