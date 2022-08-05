import sys

# Usage example (running a grammar optimized for overall efficiency):
# python3 applyCounterfactualGrammar.py English 7580379440 manual_output_funchead_two_coarse_lambda09_best_large

# Arguments
language = sys.argv[1]
model = sys.argv[2] # this can be the name of a counterfactual grammar file, or REAL_REAL (for real orderings)
BASE_DIR = sys.argv[3]


import random
from collections import deque

from math import log, exp
from random import random, shuffle
import os
import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
from corpusIterator_FuncHead import CorpusIteratorFuncHead # Located at models/corpus_reader
import numpy.random

def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x


def initializeOrderTable():
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in CorpusIteratorFuncHead(language,partition).iterator():
      for line in sentence:
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
   return depsVocab




def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   allGradients = gradients_from_the_left_sum
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients




def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       logits = [(x, distanceWeights[sentence[x-1]["dependency_key"]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = list(map(lambda x:x[0], logits))
       return childrenLinearized           

def orderSentence(sentence):
   root = None
   logProbabilityGradient = 0
   if model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         if model == "REAL_REAL":
            eliminated.append(line)
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      direction = "DH" if dhWeights[key] else "HD"
      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])

   if model == "REAL_REAL":
       while len(eliminated) > 0:
          line = eliminated[0]
          del eliminated[0]
          if "removed" in line:
             continue
          line["removed"] = True
          if "children_DH" in line:
            assert 0 not in line["children_DH"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_DH"]]
          if "children_HD" in line:
            assert 0 not in line["children_HD"]
            eliminated = eliminated + [sentence[x-1] for x in line["children_HD"]]
   else:
    for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
   if True:
     print("Original")
     print(" ".join(list(map(lambda x:x["word"], sentence)))
     print("Linarized")
     print(" ".join(list(map(lambda x:x["word"], linearized)))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   return linearized




dhWeights = {} 
distanceWeights = {} 



if model == "RANDOM": # a random ordering
   depsVocab = initializeOrderTable()
   itos_deps = sorted(vocab_deps)
   for x in itos_deps:
     dhWeights[x] = random.random() - 0.5
     distanceWeights[x] = random.random() - 0.5
elif model == "REAL_REAL":
   pass
else:
   with open("../../raw-grammars/"+BASE_DIR+"/auto-summary-lstm.tsv", "r") as inFile:
     data = [x.split("\t") for x in inFile.read().strip().split("\n")]
     header = data[0]
     data = data[1:]
    
   if "CoarseDependency" not in header:
     header[header.index("Dependency")] = "CoarseDependency"
   if "DH_Weight" not in header:
     header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
     header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
      if line[header.index("FileName") == model and line[header.index("Language") == language:
        key = line[header.index("CoarseDependency")]
        dhWeights[key] = float(line[header.index("DH_Weight")])
        distanceWeights[key] = float(line[header.index("DistanceWeight")])

corpus = CorpusIteratorFuncHead(language,"train")
corpusIterator = corpus.iterator()
for sentence in corpusIterator:
  ordered = orderSentence(sentence)
  ## DO SOMETHING WITH THE ORDERED SENTENCE HERE

