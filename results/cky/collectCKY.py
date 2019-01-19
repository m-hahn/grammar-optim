import os
path = "/home/user/CS_SCR/cky/"
with open("cky-summary.tsv", "w") as outFile:
 print >> outFile, "\t".join(["Language", "Model", "Directory", "Loss", "Total", "Gold"])
 for name in sorted(os.listdir(path)):
   with open(path+name, "r") as inFile:
      loss, total, gold = tuple(map(float, inFile.read().strip().split("\n")))
      model = name[name.index(".py")+4:name.index("_manual")]
      language, model = model[:model.rfind("_")], model[model.rfind("_")+1:]
      directory = name[name.index("manual"):-4]
      name = name[:-4].split("_")
      language = name[2]
      print >> outFile, "\t".join(map(str, [language, model, directory, loss, total, gold]))

