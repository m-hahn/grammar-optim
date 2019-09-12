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
      print(parseability, parameters)
      probabilities = tuple(x for x in parameters if x[0].startswith("prob"))
      if probabilities not in byParam:
         byParam[probabilities] = {}
      wordOrder =  dict(x for x in parameters if "_" in x[0])
      print(wordOrder)
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
      byParam[probabilities][order] = parseability
#assert False

results = []
for param in byParam:
   data = byParam[param]
   param = dict(param)
   if "Cxcomp_Axcomp_Cacl" not in data:
      continue
   Cxcomp_Axcomp_Cacl = data["Cxcomp_Axcomp_Cacl"]
   
   if "Nxcomp_Nacl" in data:
      Nxcomp_Nacl = data.get("Nxcomp_Nacl", None)
   elif float(param["probNPBranching"]) == 0.0:
      Nxcomp_Nacl = data.get("Nxcomp_Cacl", None)
   else:
      continue


   if "Cxcomp_Dxcomp_Cacl" in data:
     Cxcomp_Dxcomp_Cacl = data["Cxcomp_Dxcomp_Cacl"]
   elif float(param["probNPBranching"]) == 0.0:
     Cxcomp_Dxcomp_Cacl = data["Cxcomp_Dxcomp_Nacl"]
     assert False
   else:
     print(param["probNPBranching"])
     assert False
     continue
   improvement = Cxcomp_Dxcomp_Cacl - Nxcomp_Nacl # negative = reduction due to correlation+DLM
   results.append((improvement, param))

results = sorted(results, key=lambda x:(x[0]))
for x in results:
   print x

     
