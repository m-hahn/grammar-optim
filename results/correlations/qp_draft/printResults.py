
verb_patterner = ["adposition", "copula", "auxiliary", "noun", "noun", "complementizer", "verb", "want"]
noun_patterner = ["NP", "NP", "VP", "genitive", "relative clause", "S", "PP", "VP"]

relevant = ["lifted_case", "lifted_cop", "aux", "nmod", "acl", "lifted_mark", "obl", "xcomp"]
predicted = ["+", "+", "-", "+", "+", "+", "+", "+"]

results = {}
for typ in ["depl", "ground", "two", "langmod", "parser"]:
  with open("results-"+typ+".tsv", "r") as inFile:
    results[typ] = [x.split("\t") for x in inFile.read().strip().split("\n")]
    results[typ] = [results[typ][0], results[typ][1:]]
    results[typ][0] = dict(zip(results[typ][0], range(len(results[typ][0]))))
    results[typ][1] = [[y[0]] + [float(x) for x in y[1:]] for y in results[typ][1]]

relations = [x[0] for x in results["depl"][1]]
relations = dict(zip(relations, range(len(relations))))

def getStars(x):
   if x > 0.05:
     return ""
   if x > 0.01:
     return "$^{*}$"
   if x > 0.001:
     return "$^{**}$"
   return "$^{***}$"


def extractEffect(typ, row, direction):
   depl = results[typ][1][row][results[typ][0]["satisfiedFraction"]]
   deplP = results[typ][1][row][results[typ][0]["posteriorOpposite"]]
   if deplP > 0.5 and depl < 0.5:
      deplP = 1-deplP
   if direction == "-":
      depl = 1-depl
   depl = str(round(depl*100))
   if deplP < 0.05:
      depl = "\\textbf{"+depl+"}"
   return depl, deplP

def prettyPrint(x):
   return x.replace("_", "\\_")

for v, n, rel, direction in zip(verb_patterner, noun_patterner, relevant, predicted):
   row = relations[rel]
   ground = results["ground"][1][row][results["ground"][0]["satisfiedFraction"]]
   depl, deplP = extractEffect("depl", row, direction)
   langmod, langmodP = extractEffect("langmod", row, direction)
   parser, parserP = extractEffect("parser", row, direction)
   two, twoP = extractEffect("two", row, direction)
   print("".join(map(str, [ v, "  &  ", n, "  &  ", prettyPrint(rel), "  &  ", round(ground*100) , "   &   ", depl, getStars(deplP) , "   &   ", langmod, getStars(langmodP), "   &   ", parser, getStars(parserP), "   &   ", two, getStars(twoP) , "   \\\\"])))



print("\n")
print("\n")

for rel in (relations):
   row = relations[rel]
   direction = "+"
   ground = results["ground"][1][row][results["ground"][0]["satisfiedFraction"]]
   depl, deplP = extractEffect("depl", row, direction)
   langmod, langmodP = extractEffect("langmod", row, direction)
   parser, parserP = extractEffect("parser", row, direction)
   two, twoP = extractEffect("two", row, direction)
   
   print("".join(map(str, [ prettyPrint(rel), "  &  ", round(ground*100) , "   &   ", depl, getStars(deplP) , "   &   ", langmod, getStars(langmodP), "   &   ", parser, getStars(parserP), "   &   ", two, getStars(twoP) , "   \\\\"])))



