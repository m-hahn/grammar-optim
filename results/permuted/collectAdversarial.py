import os


path = "CS_SCR/deps/ADVERSARIAL_PARSER/"

files = os.listdir(path)

with open("adversarial-parser.tsv", "w") as outFile:
 print >> outFile, "\t".join(["Language", "Type", "Model", "FileName", "Loss", "UAS"])
 for name in files:
   with open(path+name, "r") as inFile:
       data = [x.split(" ") for x in inFile.read().strip().split("\n")]
       _ = data[0]
       _ = data[1]
       loss = data[2]
       if len(loss) == 1 or float(loss[-1]) < float(loss[-2]):
           continue
       uas = data[3]
       las = data[4]
       commands = data[5]

       language = commands[1]
       ordering = commands[-1]
       model = commands[-2]
       modelNum = commands[-3]
       if ordering == "pos" and "learn" not in name:
           continue
       if ordering == "pos":
           ordering = "lexicographic"
#       print(ordering)
       loss = loss[-2]
       uas = uas[-2]
       print >> outFile, "\t".join([language, ordering, model, modelNum, loss, uas])
