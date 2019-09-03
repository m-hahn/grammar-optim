#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"



# TODO also try other optimizers

import random
import sys

objectiveName = "LM"

language = sys.argv[1]

with open("../memory-surprisal/code/branching_entropy/branching_entropy_coarse_byRelation.tsv", "r") as inFile:
   branchingEntropies = [x.split("\t") for x in inFile.read().split("\n")]
   header = branchingEntropies[0]
   header = dict(zip(header, range(len(header))))
   branchingEntropies = branchingEntropies[1:]
   branchingEntropies = [x for x in branchingEntropies if x[header["Language"]] == language]
   assert len(branchingEntropies) > 0
branchingEntropies = dict([(x[1], float(x[2])) for x in branchingEntropies])
print(branchingEntropies)

from math import log

def f(x):
   return -(x*log(x) + (1-x) * log(1-x))

branchingDeterministicProbabilities = {}

for dep, ent in branchingEntropies.iteritems():
 # print(ent)
  upper = 1.0
  lower = 0.5
  while abs(upper-lower) > 0.0001:
    mean = (upper+lower)/2
    if f(mean) < ent:
       upper = mean
    else:
       lower = mean
#  print(lower, upper, f(lower), ent)
  branchingDeterministicProbabilities[dep] = (lower+upper)/2
  print("\t".join(map(str,[dep, "PROB DET", branchingDeterministicProbabilities[dep], ent])))
#quit()


