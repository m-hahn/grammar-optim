import os

path = "/u/scr/mhahn/deps/manual_output_funchead_two_coarse_lambda09/"

files = os.listdir(path)

cache = {}

def extractModelTypeCached(modelName, objName):
   if (modelName, objName) not in cache:
     cache[(modelName,objName)] = extractModelType(modelName,objName)
   return cache[(modelName,objName)]
   

def extractModelType(modelName, objName):
   if modelName == "readDataDistCrossGPUFreeAllTwoEqual.py":
      return "Two"
   print ["UNKNOWN TYPE", modelName]
   return modelName


inHeader = ["FileName", "Counter", 'AverageLoss_LM', 'AverageLAS', 'AverageLoss_Parser', 'AverageUAS',  "DH_Weight", "CoarseDependency", "DistanceWeight"] #, "EntropyWeight", "ObjectiveName", "LR_POLICY", "Lagrange_Lambda", "Lagrange_B", "L2_Weight", 'DH_Sigma', 'Distance_Mean', 'Distance_Sigma', 'DH_Mean', 'Var_Slope_DepLength', 'Mean_Slope_DepLength', 'Mean_Slope_Surp_POS', 'Var_Slope_Surp_POS', 'POS', 'Var_Slope_Surp_Word', 'Mean_Slope_Surp_Word', 'Var_Slope_UID', 'Mean_Slope_UID', 'Mean_Slope_UIDRate', 'LogVar_Slope_Surp_POS', 'LogVar_Slope_Surp_Word', 'LogVar_Slope_UIDRate', 'LogVar_Slope_DepLogLength', 'Distance_LogSigma', 'Mean_Slope_DepLogLength', 'DH_LogSigma', 'LogVar_Slope_DepLength', 'LogVar_Slope_UID', 'Mean_Slope_UIDRate_rateVariance', 'LogVar_Slope_UIDRate_plainVariance', 'Mean_Slope_UIDRate_rateL1Divergence', 'LogVar_Slope_UIDRate_successiveL1', 'Mean_Slope_UIDRate_successiveL1', 'LogVar_Slope_UIDRate_rateVariance', 'Mean_Slope_UIDRate_plainL1Divergence', 'Mean_Slope_UIDRate_plainVariance', 'Mean_Slope_UIDRate_successiveL1Rate', 'LogVar_Slope_UIDRate_plainL1Divergence', 'LogVar_Slope_UIDRate_successiveL1Rate', 'LogVar_Slope_UIDRate_rateL1Divergence', 'LogExpVar_Slope_Surp_POS', 'LogExpVar_Slope_Surp_Word', 'DevLikelihood', 'Distance_LogExpSigma', 'DH_LogExpSigma', 'LogExpVar_Slope_DepLength', 'LogExpVar_Slope_UID', 'LogExpVar_Slope_UIDRate', 'D_plainVariance_DH_Weight', 'D_successiveL1Rate_DH_Weight', 'D_successiveL1_DH_Weight', 'D_UIDplainL1Divergence_DistanceWeight', 'D_UIDrateVariance_DistanceWeight', 'D_plainL1Divergence_DH_Weight', 'D_UIDrateL1Divergence_DistanceWeight', 'D_rateVariance_DH_Weight', 'D_UIDplainVariance_DistanceWeight', 'D_rateL1Divergence_DH_Weight', 'D_UIDsuccessiveL1_DistanceWeight', 'D_UIDsuccessiveL1Rate_DistanceWeight', 'D_DepL_DH_Weight', 'D_DepL_DistanceWeight', 'D_WordSurp_DistanceWeight', 'Dummy', 'D_WordSurp_DH_Weight', "MI", 'D_WordSurp_DistanceWeight_HD', 'D_WordSurp_DistanceWeight_DH', 'DistanceHD_Sigma', 'DistanceDH_Mean', 'DistanceHD_Mean', 'DistanceDH_Sigma', 'Distance_Mean_NoPunct', 'Distance_Sigma_NoPunct', 'DH_Sigma_NoPunct', 'DH_Mean_NoPunct', 'DependentLength_Median', 'MeanDependentLength', 'DependentLength_FirstQuartile', 'DependentLength_ThirdQuartile']
outHeader = ["Language"] + inHeader



with open(path+"auto-summary-lstm.tsv", "w") as outFile:
  print >> outFile, "\t".join(outHeader) #, "FileName", "ModelName", "Counter", "AverageLoss", "Head", "DH_Weight", "Dependency", "Dependent", "DistanceWeight"])
  for filename in files:
     if "model" in filename:
        print "READING "+filename 
        part1 = filename.split("_model_")[0]
        if "_" in part1:
          language = part1.split("_")[0]
        else:
          language = "English"
        with open(path+filename, "r") as inFile:
            try:
              header = next(inFile).strip().split("\t")
            except StopIteration:
              print ["EMPTY FILE?",path+filename]
              continue
            missingColumns = len(inHeader) - len(header)
#            assert missingColumns >= 0, [inHeader, header, set(header) - set(inHeader)]
            
            for i in range(len(header)):
              if header[i] in ["AverageLength", "Perplexity"]:
                  header[i] = "AverageLoss"
            if len(set(header) - set(inHeader)) > 0:
              print set(header) - set(inHeader)
#            matched = zip(inHeader+([None]*missingColumns), header)          
 #           print filter(lambda x:x[0] != x[1], matched)
            lineCounter = 0
            if "Pukwac" in filename:
               language = "Pukwac" 

            for line in inFile:
               lineCounter += 1
               line = line.strip().split("\t")
               outLine = [language] #, extractModelType(line[1])]
               for colName in inHeader:
                  if colName == "Language" and "Pukwac" in filename:
                      outLine.append("Pukwac")
                      continue
                  try:
                    i = header.index(colName)
                    outLine.append(line[i])
                    if outLine[-1].startswith("["):
                        outLine[-1] = outLine[-1].replace("[","").replace("]","")
                    if colName == "ObjectiveName":
                       if line[i] != extractModelTypeCached(line[1], line[i]) and lineCounter == 1:
                          print [line[i], "CHOOSING INSTEAD", extractModelTypeCached(line[1], line[i])]
                       outLine[-1] = extractModelTypeCached(line[1], line[i])
                  except ValueError:
                    if colName == "ObjectiveName":
                       outLine.append(extractModelTypeCached(line[1], "NONE"))
                    else:
                       outLine.append("NA")
#                    print colName
#               outLine += (["NA"]*missingColumns)
               assert len(outLine) == len(outHeader)
               print >> outFile, "\t".join(outLine)



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

