import os
PATH = "../../../raw-results/recoverability-toy-simulation/"
files= os.listdir(PATH)
byParam = {}
for name in files:
   with open(PATH+name, "r") as inFile:
      parseability = float(next(inFile).split("\t")[1])
      next(inFile)
      next(inFile)
      parameters = sorted([tuple(x.split("=")) for x in next(inFile).strip().split(" ")])
      #print(parseability, parameters)
      probabilities = tuple(x for x in parameters if x[0].startswith("prob"))
      if probabilities not in byParam:
         byParam[probabilities] = {}
      wordOrder =  dict(x for x in parameters if "_" in x[0])
      #print(wordOrder)
      if wordOrder["correlation_xcomp"] == "True" and wordOrder["dlm_xcomp"] == "True":
         order = "Cxcomp_Dxcomp"
      elif wordOrder["correlation_xcomp"] == "True" and wordOrder["dlm_xcomp"] == "False":
         order = "Cxcomp_Axcomp"
      elif wordOrder["correlation_xcomp"] == "False":
         order = "Nxcomp"
      if wordOrder["correlation_acl"] == "True":
         order += "_Cacl"
      else:
         order += "_Nacl"
      if order not in byParam[probabilities]:
         byParam[probabilities][order] = []
      byParam[probabilities][order].append(parseability)
#assert False

results = []
for param in byParam:
   data = byParam[param]
   param = dict(param)
   
   if "Nxcomp_Nacl" in data:
      Nxcomp_Nacl = data["Nxcomp_Nacl"]
   elif float(param["probNPBranching"]) == 0.0 and "Nxcomp_Cacl" in data:
      Nxcomp_Nacl = data["Nxcomp_Cacl"]
   else:
      continue


   if "Cxcomp_Dxcomp_Cacl" in data:
     Cxcomp_Dxcomp_Cacl = data["Cxcomp_Dxcomp_Cacl"]
   elif float(param["probNPBranching"]) == 0.0 and "Cxcomp_Dxcomp_Nacl" in data:
     Cxcomp_Dxcomp_Cacl = data["Cxcomp_Dxcomp_Nacl"]
     assert False
   else:
     print("Missing", param["probNPBranching"])
     #assert False
     continue
   assert Cxcomp_Dxcomp_Cacl is not None
   assert Nxcomp_Nacl is not None
   Cxcomp_Dxcomp_Cacl = sum(Cxcomp_Dxcomp_Cacl) / len(Cxcomp_Dxcomp_Cacl)
   Nxcomp_Nacl = sum(Nxcomp_Nacl) / len(Nxcomp_Nacl)
   improvement = Cxcomp_Dxcomp_Cacl - Nxcomp_Nacl # negative = reduction due to correlation+DLM
   results.append((improvement, param))

results = sorted(results, key=lambda x:(x[0]))
with open("results.tsv", "w") as outFile:
   print >> outFile, ("\t".join([str(x) for x in ["Diff", "probVPBranching", "probNPBranching", "probObj"]]))
   for x in results:
   #   print x
      diff = x[0]
      probVPBranching = x[1]["probVPBranching"]
      probNPBranching = x[1]["probNPBranching"]
      probObj = x[1]["probObj"]
      print >> outFile, ("\t".join([str(x) for x in [diff, probVPBranching, probNPBranching, probObj]]))
   

     
