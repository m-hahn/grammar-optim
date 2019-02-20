import random

processedGrammars = set()

assignment = {}

with open("auto-summary-lstm.tsv", "r") as inFile:
    header = next(inFile).strip().split("\t")
    header = dict(zip(header, range(len(header))))
    for line in inFile:
        line = line.strip().split("\t")
        language = line[header["Language"]]
        if language == "Ancient":
            language = "Ancient_Greek"
        if language == "Old":
            language = "Old_Church_Slavonic"
        fileName = line[header["FileName"]]
        if (language, fileName) in processedGrammars:
            continue
        CoarseDependency = line[header["CoarseDependency"]]
        if CoarseDependency == "obj":
           DH_Weight = float(line[header["DH_Weight"]])
#           print(DH_Weight)
           if str(DH_Weight) == "nan":
               continue
           assert DH_Weight > 5 or DH_Weight < -5
           direction = ("OV" if DH_Weight > 0 else "VO")
           if (language, direction) not in assignment:
               assignment[(language, direction)] = []
           assignment[(language, direction)].append([language, fileName, direction])

with open("model-partition.tsv", "w") as outFile:
    print >> outFile, "\t".join(["Language", "FileName", "Direction", "Group"])
    random.seed(45)
    for key in sorted(assignment.keys()):
     #   print(assignment[key])
        random.shuffle(assignment[key])
        for i in range(len(assignment[key])):
            assignment[key][i].append("GROUP"+str(i%4+1))
            print >> outFile, ("\t".join(assignment[key][i]))
    #    print(assignment[key])
        
    #print(assignment)
    
    
