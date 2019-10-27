

patterns=[]
patterns.append(("adposition","NP","lifted_case"))
patterns.append(("copula","NP","lifted_cop"))
patterns.append(("auxiliary","VP","aux"))
patterns.append(("noun","genitive","nmod"))
patterns.append(("noun","relative clause","acl"))
patterns.append(("complementizer","S","lifted_mark"))
patterns.append(("verb","PP","obl"))
patterns.append(("want","VP","xcomp"))

def tableByFirst(x):
    header = x[0]
    body = dict((z[0], z) for z in x[1:])
    return (header, body)
with open("results-ground-agree.tsv", "r") as inFile:
    ground = tableByFirst([x.split("\t") for x in inFile.read().strip().split("\n")])
with open("results-prevalence-depl.tsv", "r") as inFile:
    depl = tableByFirst([x.split("\t") for x in inFile.read().strip().split("\n")])
with open("results-prevalence-two-09-best.tsv", "r") as inFile:
    efficiency = tableByFirst([x.split("\t") for x in inFile.read().strip().split("\n")])
with open("results-prevalence-langmod-best.tsv", "r") as inFile:
    surp = tableByFirst([x.split("\t") for x in inFile.read().strip().split("\n")])
with open("results-prevalence-parser-best.tsv", "r") as inFile:
    parse = tableByFirst([x.split("\t") for x in inFile.read().strip().split("\n")])

def getStars(mean, posterior):
   posterior = float(posterior)
   if posterior > 0.5:
       posterior = 1-posterior
   if posterior < 0.001:
       return "\\textbf{"+mean+"}$^{***}$"
   if posterior < 0.01:
       return "\\textbf{"+mean+"}$^{**}$"
   if posterior < 0.05:
       return "\\textbf{"+mean+"}$^{*}$"
   return mean

def convert(dep, mean):
    if dep == "aux":
        return 100-mean
    else:
        return mean

for verb, obj, dep in patterns:
   result = [verb, obj, dep.replace("_", "\_")]
   lineGround = ground[1][dep]
   lineDepL = depl[1][dep]
   linePred = surp[1][dep]
   linePars = parse[1][dep]
   lineEff = efficiency[1][dep]
   result.append(str(int(round(convert(dep,100*float(lineGround[ground[0].index("satisfiedFraction")]))))))
   result.append(str(int(round(convert(dep,100*float(lineDepL[depl[0].index("satisfiedFraction")]))))))
   result[-1] = getStars(result[-1], lineDepL[depl[0].index("posteriorOpposite")])
   result.append(str(int(round(convert(dep,100*float(linePred[surp[0].index("satisfiedFraction")]))))))
   result[-1] = getStars(result[-1], linePred[surp[0].index("posteriorOpposite")])
   result.append(str(int(round(convert(dep,100*float(linePars[parse[0].index("satisfiedFraction")]))))))
   result[-1] = getStars(result[-1], linePars[parse[0].index("posteriorOpposite")])
   result.append(str(int(round(convert(dep,100*float(lineEff[efficiency[0].index("satisfiedFraction")]))))))
   result[-1] = getStars(result[-1], lineEff[efficiency[0].index("posteriorOpposite")])
   result[-1] += "   \\\\"
   print("    &    ".join(result))

