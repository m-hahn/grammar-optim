import random
import sys

objectiveName = "cky"

print sys.argv

language = sys.argv[1]
model = sys.argv[2]
BASE_DIR = sys.argv[3]

FILE_NAME = "cky"


myID = random.randint(0,10000000)


posUni = set()
posFine = set()




from math import log, exp, sqrt
from random import random, shuffle, randint
import os


from corpusIterator_FuncHead import CorpusIteratorFuncHead

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
     for sentence in CorpusIteratorFuncHead(language,partition).iterator():
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
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))
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

   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   if model == "REAL_REAL":
      eliminated = []
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"):
         if model == "REAL_REAL":
            eliminated.append(line)
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + exp(-dhLogit))
      dhSampled = (0.5 < probability)
      line["ordering_decision_log_probability"] = 0 #torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability)+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])



   if model != "REAL_REAL":
      for line in sentence:
         if "children_DH" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
            line["children_DH"] = childrenLinearized
         if "children_HD" in line:
            childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
            line["children_HD"] = childrenLinearized
   else:
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


   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if model == "REAL_REAL":
      linearized = filter(lambda x:"removed" not in x, sentence)
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
   dhLogits[key] = 100 * 2*(random()-0.5)
   dhWeights[i] = dhLogits[key]

   originalDistanceWeights[key] = 100 * random()  
   distanceWeights[i] = originalDistanceWeights[key]

import os

if model != "RANDOM" and model != "REAL_REAL":
   temperature = 1.0
   inpModels_path = "/u/scr/mhahn/deps/"+"/"+BASE_DIR+"/"
   models = os.listdir(inpModels_path)
   models = filter(lambda x:"_"+model+".tsv" in x, models)
   if len(models) == 0:
     assert False, "No model exists"
   if len(models) > 1:
     assert False, [models, "Multiple models exist"]
   
   with open(inpModels_path+models[0], "r") as inFile:
      data = map(lambda x:x.split("\t"), inFile.read().strip().split("\n"))
      header = data[0]
      data = data[1:]
    
   if "Dependency" not in header:
      header[header.index("CoarseDependency")] = "Dependency"
   if "DH_Weight" not in header:
      header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
      header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
      dependency = line[header.index("Dependency")]
      key = dependency
      dhWeights[stoi_deps[key]] = temperature*float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = temperature*float(line[header.index("DistanceWeight")])






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
for sentence in CorpusIteratorFuncHead(language,partition).iterator():
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

totalRootCount = sum([roots[x] for x in roots])


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

       chart = [[[None for _ in itos_pos_uni] for _ in posString] for _ in posString]

       backpointers = [[[None for _ in itos_pos_uni] for _ in posString] for _ in posString]


       for length in range(1, len(posString)+1): # the NUMBER of words spanned. start+length is the first word OUTSIDE the constituent
          for start in range(len(posString)): # the index of the first word taking part in the thing
             if start+length-1 >= len(posString):
                continue
             if length == 1:
                  assert start == start+length-1
                  chart[start][start][stoi_pos_uni[posString[start]]] = 0.0
             else:
                 for start2 in range(start+1, len(posString)):
                   for ipos1, pos1 in enumerate(itos_pos_uni): # for the left
                      for ipos2, pos2 in enumerate(itos_pos_uni): # for the right
                         countsR = log(productions[(pos1, pos2, "r")]) - log(headCount[pos1])
                         countsL = log(productions[(pos2, pos1, "l")]) - log(headCount[pos2])

                         left = chart[start][start2-1][ipos1]
                         right = chart[start2][start+length-1][ipos2]
                         if left is None or right is None:
                            continue

                         newR = countsR + left + right
                         newL = countsL + left + right
                         entryR = chart[start][start+length-1][ipos1]
                         chart[start][start+length-1][ipos1] = logSumExp(newR, entryR)

                         entryL = chart[start][start+length-1][ipos2]
                         chart[start][start+length-1][ipos2] = logSumExp(newL, entryL)

       for ipos, pos in enumerate(itos_pos_uni):
           if chart[0][-1][ipos] is not None:
              chart[0][-1][ipos] += log(roots[pos]) - log(totalRootCount)



       goldProbability = 0

       childrenProbsLeft = [[] for _ in batchOrdered[0]]
       childrenProbsRight = [[] for _ in batchOrdered[0]]

       for i, word in enumerate(batchOrdered[0]):
          dep = word["coarse_dep"]
          pos = posString[i]
          assert pos == word["posUni"]
          if dep == "root":
            goldProbability  +=  log(roots[pos]) - log(totalRootCount)
          else:
            head = word["reordered_head"]
            direction = ("l" if i+1 < head else "r")
            head_pos = batchOrdered[0][head-1]["posUni"]
            (childrenProbsLeft if direction == "l" else childrenProbsRight)[head-1].append(log(productions[(head_pos, pos, direction)]) - log(headCount[head_pos]))
       
       for i in range(len(batchOrdered[0])):
          a = len(childrenProbsLeft[i])
          b = len(childrenProbsRight[i])
          l1 = sum([log(x) for x in range(1, a+b+1)])
          l2 = sum([log(x) for x in range(1, a+1)])
          l3 = sum([log(x) for x in range(1, b+1)])

          goldProbability += sum(childrenProbsLeft[i]) + sum(childrenProbsRight[i]) + l1 - l2 - l3



#       print(chart[0][-1])
       fullProb = log(sum([exp(x) if x is not None else 0 for x in chart[0][-1]]))
       goldProb = goldProbability
       conditional = (fullProb - goldProb)
       return conditional, len(batchOrdered[0]), fullProb, goldProb

corpusDev = CorpusIteratorFuncHead(language, "dev")

conditionalTotal = 0
marginalTotal = 0
goldTotal = 0
lengthTotal = 0

for i, sentence in enumerate(corpusDev.iterator(rejectShortSentences = True)):
    conditional, length, marginal, gold = forward([sentence])
    conditionalTotal += conditional
    marginalTotal += marginal
    goldTotal += gold
    lengthTotal += length
    print(language, i, conditionalTotal/lengthTotal, marginalTotal/lengthTotal, goldTotal/lengthTotal)
    if i > 500:
       break
with open("/u/scr/mhahn/cky/"+__file__+"_"+language+"_"+model+"_"+BASE_DIR+".txt", "w") as outFile:
   print >> outFile, conditionalTotal/lengthTotal
   print >> outFile, marginalTotal/lengthTotal
   print >> outFile, goldTotal/lengthTotal



