import os


path = "/u/scr/mhahn/deps/language_modeling_adversarial/"

files = os.listdir(path)

with open("adversarial-lm.tsv", "w") as outFile:
 print >> outFile, "\t".join(["Language", "Type", "Model", "FileName", "LM_Loss"])
 for name in files:
   with open(path+name, "r") as inFile:
       data = [x.split("\t") for x in inFile.read().strip().split("\n")]
       language = data[0][1]
       ordering = data[0][-1]
       Model = data[0][-2]
       loss = data[1]
       if len(loss) == 1 or float(loss[-1]) < float(loss[-2]):
           continue
       loss = data[2]
       loss = loss[-2]
       filename = data[0][-3]
       if ordering == "pos":
           ordering = "lexicographic"
       print >> outFile, "\t".join([language, ordering, Model, filename, loss])
