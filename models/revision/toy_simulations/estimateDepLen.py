import random
import sys

objectiveName = "graphParser"

print sys.argv

language = "obj_xcomp_acl" #sys.argv[1]
model = "mod" #sys.argv[21]


import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--correlation_xcomp', default=True, type=lambda x: (str(x).lower() == 'true')) # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
parser.add_argument('--dlm_xcomp', default=True, type=lambda x: (str(x).lower() == 'true')) # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
parser.add_argument('--correlation_acl', default=True, type=lambda x: (str(x).lower() == 'true')) # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
parser.add_argument('--probVPBranching', default=0.5, type=float) 
parser.add_argument('--probObj', default=0.5, type=float) 
parser.add_argument('--probNPBranching', default=0.0, type=float) 

args=parser.parse_args()




myID = random.randint(0,10000000)


posUni = set()
posFine = set()




from math import log, exp, sqrt
from random import random, shuffle, randint
import os


import corpusIterator_Toy


corpusIterator_Toy.probVPBranching = args.probVPBranching
corpusIterator_Toy.probObj         = args.probObj        
corpusIterator_Toy.probNPBranching = args.probNPBranching


originalDistanceWeights = {}


def makeCoarse(x):
   if ":" in x:
      return x[:x.index(":")]
   return x

def initializeOrderTable():
   orderTable = {}
   keys = set()
   vocab = {}
   distanceSum = {}
   distanceCounts = {}
   depsVocab = set()
   for partition in ["train", "dev"]:
     for sentence in corpusIterator_Toy.dev():
      for line in sentence:
          vocab[line["word"]] = vocab.get(line["word"], 0) + 1
          line["coarse_dep"] = makeCoarse(line["dep"])
          depsVocab.add(line["coarse_dep"])
          posFine.add(line["posFine"])
          posUni.add(line["posUni"])
  
          if line["coarse_dep"] == "root":
             continue
          posHere = line["posUni"]
          posHead = sentence[line["head"]-1]["posUni"]
          dep = line["coarse_dep"]
          direction = "HD" if line["head"] < line["index"] else "DH"
          key = dep
          keyWithDir = (dep, direction)
          orderTable[keyWithDir] = orderTable.get(keyWithDir, 0) + 1
          keys.add(key)
          distanceCounts[key] = distanceCounts.get(key,0.0) + 1.0
          distanceSum[key] = distanceSum.get(key,0.0) + abs(line["index"] - line["head"])
   dhLogits = {}
   for key in keys:
      hd = orderTable.get((key, "HD"), 0) + 1.0
      dh = orderTable.get((key, "DH"), 0) + 1.0
      dhLogit = log(dh) - log(hd)
      dhLogits[key] = dhLogit
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


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

import numpy.random
import numpy as np

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()
logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]] for x in remainingChildren]
           if reverseSoftmax:
              logits = [-x for x in logits]
           softmax = logits #.view(1,-1).view(-1)
           selected = numpy.argmax(softmax)
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           


def orderSentence(sentence, dhLogits, printThings):
   global model
#   print(sentence)
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + exp(-dhLogit))
      dhSampled = (0.5 < probability)
      line["ordering_decision_log_probability"] = 0 

      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability)+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])


   if True:
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
      x["reordered_index"] = 1+i
      if x["head"] == 0: # root
         x["reordered_head"] = 0
      else:
         x["reordered_head"] = 1+moved[x["head"]-1]
   assert len(linearized) == len(sentence), (len(linearized), len(sentence))
   #assert linearized[0]["word"] == "v", linearized
   return linearized, logits


dhLogits, vocab, vocab_deps, depsVocab = initializeOrderTable()

posUni = list(posUni)
itos_pos_uni = posUni
stoi_pos_uni = dict(zip(posUni, range(len(posUni))))

posFine = list(posFine)
itos_pos_ptb = posFine
stoi_pos_ptb = dict(zip(posFine, range(len(posFine))))



itos_pure_deps = sorted(list(depsVocab)) 
stoi_pure_deps = dict(zip(itos_pure_deps, range(len(itos_pure_deps))))
   

itos_deps = sorted(vocab_deps)
stoi_deps = dict(zip(itos_deps, range(len(itos_deps))))

print itos_deps

dhWeights = [0.0] * len(itos_deps)
distanceWeights = [0.0] * len(itos_deps)
for i, key in enumerate(itos_deps):
   dhLogits[key] = 2*(random()-0.5)
   dhWeights[i] = dhLogits[key]

   originalDistanceWeights[key] = random()  
   distanceWeights[i] = originalDistanceWeights[key]


   if key == "xcomp":
     dhWeights[i] = -10.0 if args.correlation_xcomp else 10.0
     print(args.correlation_xcomp, dhWeights[i])
     distanceWeights[i] = 10.0 if args.dlm_xcomp else 0.5
   elif key == "obj":
     dhWeights[i] = -10.0 
     distanceWeights[i] = 1.0
   elif key == "acl":
     dhWeights[i] = -10.0 if args.correlation_acl else 10.0
     distanceWeights[i] = 10.0
   else:
     assert False, key
print(args, itos_deps)    
print(dhWeights)
#quit()

import os





words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))


if len(itos) > 6:
   assert stoi[itos[5]] == 5


# create table of transititions

roots = {}
productions = {} # keys: head, dependent, direction. values: count
headCount = {}

# Dirichlet smoothing
for hd in itos_pos_uni:
   roots[hd] = 1.0
   for dp in itos_pos_uni:
     for lr in "lr":
        productions[(hd,dp,lr)] = 1.0
   headCount[hd] = 1.0
orderTable = {}
keys = set()
vocab = {}
distanceSum = {}
distanceCounts = {}
depsVocab = set()
partition = "train"
for sentence in corpusIterator_Toy.training():
   for line in sentence:
       line["coarse_dep"] = makeCoarse(line["dep"])
       posHere = line["posUni"]
       if line["coarse_dep"] == "root":
          roots[posHere] += 1
          continue
       posHead = sentence[line["head"]-1]["posUni"]
       dep = line["coarse_dep"]
       direction = "l" if (dhWeights[stoi_deps[dep]] > 0.5) else "r"
#       direction = "r" if line["head"] < line["index"] else "l"
       productions[(posHead, posHere, direction)] += 1
       headCount[posHead] += 1



print(productions)
#for x in productions:
#   productions[x] = 0
#productions[('v', 'n', 'l')] = args.
#productions[('n', 'v', 'l')] = 
#productions[('v', 'v', 'l')] = 

#{: 224829.0, ('n', 'v', 'l'): 24659.0, ('v', 'v', 'r'): 1.0, ('v', 'v', 'l'): 56278.0, ('n', 'v', 'r'): 1.0, ('v', 'n', 'l'): 1.0, ('n', 'n', 'r'): 1.0, ('n', 'n', 'l'): 1.0}


print(list(productions))

totalRootCount = sum([roots[x] for x in roots])
print(totalRootCount)
print(roots)

print("===")

producingTerminalProbability = {}

botCountForNonTerminal = {}

for head in itos_pos_uni:
    count_top = 0
    count_bot = 0
#    count_bot += roots[head]
    for prod, prod_count in productions.iteritems():
        # Is a parent
        if prod[0] == head:
           count_top += prod_count

        # Is a child
        if prod[0] == head:
           count_bot += prod_count
        if prod[1] == head:
           count_bot += prod_count
    count_bot += roots[head]
    producingTerminalProbability[head] = 1 - count_top / count_bot
    botCountForNonTerminal[head] = count_bot
    print((head, count_top, count_bot, count_bot - count_top, roots[head], count_top / count_bot))
print(botCountForNonTerminal)
#quit()
print(producingTerminalProbability)

from torch import optim


def prod(x):
   r = 1
   for s in x:
     r *= s
   return r

crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#import torch.cuda
import torch.nn.functional

from math import exp

def logSumExp(x,y):
   if x is None:
     return y
   if y is None:
     return x
   constant = max(x,y)
   return constant + log(exp(x-constant) + exp(y-constant))

def forward(current, computeAccuracy=False, doDropout=True):
       
       printHere = False
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
       assert len(batchOrdered) == 1
  
        
       posString = [x["posUni"] for x in batchOrdered[0]]

       realHeads = [x["reordered_head"] for x in batchOrdered[0]]
       numberOfDependencies = len(realHeads)-1
       totalDepLen = 0
       for i in range(len(realHeads)):
          if realHeads[i] != 0:
              totalDepLen += abs(i+1 - realHeads[i])
       return totalDepLen, numberOfDependencies

corpusDev = corpusIterator_Toy.dev()

totalDepLen_ = 0.0
numberOfDependencies_ = 0
numberOfSentences_ = 0
for i, sentence in enumerate(corpusDev):
    numberOfSentences_ += 1
    totalDepLen, numberOfDependencies = forward([sentence])
    numberOfDependencies_ += numberOfDependencies
    totalDepLen_ += totalDepLen
    if i % 100 == 0:
       print(language, i, totalDepLen_/(numberOfDependencies_+1e-9), totalDepLen_/(numberOfSentences_+1e-9))
FILE_NAME = __file__
with open("../../../raw-results/recoverability-toy-simulation/deplen-"+language+"_"+FILE_NAME+"_model_"+str(myID)+".txt", "w") as outFile:
     print >> outFile, "DepLenPerDep"+"\t"+str(totalDepLen_/numberOfDependencies_)
     print >> outFile, "DepLenPerSent"+"\t"+str(totalDepLen_/numberOfSentences_)
     print >> outFile, (str(args).replace("Namespace(", "")[:-1].replace(", ", " "))
  

#    if i > 500:
 #      break
#with open("/u/scr/mhahn/cky/"+__file__+"_"+language+"_"+model+"_"+BASE_DIR+".txt", "w") as outFile:
#   print >> outFile, conditionalTotal/lengthTotal
#   print >> outFile, marginalTotal/lengthTotal
#   print >> outFile, goldTotal/lengthTotal



