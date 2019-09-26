# Fits grammars to orderings found in corpora.
# Michael Hahn, 2019

import random
import sys

import torch.nn as nn
import torch
from torch.autograd import Variable
import math



import os

language = sys.argv[1]

myID = random.randint(0,10000000)

posUni = set()
posFine = set()



from math import log, exp
from random import random, shuffle

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

import os
sys.path.append("..")
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
   #print orderTable
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
   # Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum

   largestDepth = 0
   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         d2 = recursivelyLinearize(sentence, child, result, allGradients)
         largestDepth = max(d2, largestDepth)
   result.append(line)
   if "children_HD" in line:
      for child in line["children_HD"]:
         d2 = recursivelyLinearize(sentence, child, result, allGradients)
         largestDepth = max(d2, largestDepth)
   return largestDepth + 1

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [0 for x in remainingChildren]
           selected = 0
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           



def orderSentence(sentence, printThings=False):
   root = None
   logits = [None]*len(sentence)
   logProbabilityGradient = 0
   for line in sentence:
      line["coarse_dep"] = makeCoarse(line["dep"])
      if line["coarse_dep"] == "root":
          root = line["index"]
          continue
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhSampled = True
   
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8]     ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])


      
   for line in sentence:
      arity.append(len(line.get("children_DH", [])) + len(line.get("children_HD", [])))
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized

   
   linearized = []
   tree_depth.append(recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0]))))
   if printThings or len(linearized) == 0:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))


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

#print itos_deps





words = list(vocab.iteritems())
#print(words)

totalCount = sum(x[1] for x in words)
probs = [float(x[1])/totalCount for x in words]
unigram_entropy = -sum([x*log(x) for x in probs])
#print(unigram_entropy)

sentenceLengths = []

tree_depth = []
arity = []

numberOfSentences = 0

for sentence in CorpusIteratorFuncHead(language,"train").iterator():
   orderSentence(sentence)
   sentenceLengths.append(len(sentence))
   numberOfSentences += 1
#print(sentenceLengths)
#print(arity)
#print(tree_depth)

def median(x):
   return sorted(x)[int(len(x)/2)]
def mean(x):
   return float(sum(x))/len(x)

with open("results/properties-"+language, "w") as outFile:
   print >> outFile, ("UnigramEntropy\t"+str(unigram_entropy))
   print >> outFile, ("SentenceCount\t"+str(numberOfSentences))
   print >> outFile, ("MedianArity\t"+str(median(arity)))
   print >> outFile, ("MedianTreeDepth\t"+str(median(tree_depth)))
   print >> outFile, ("MeanArity\t"+str(mean(arity)))
   print >> outFile, ("MeanTreeDepth\t"+str(mean(tree_depth)))
   print >> outFile, ("MaxArity\t"+str(max(arity)))
   print >> outFile, ("MaxTreeDepth\t"+str(max(tree_depth)))
   print >> outFile, ("MedianSentenceLength\t"+str(median(sentenceLengths)))
   print >> outFile, ("MeanSentenceLength\t"+str(mean(sentenceLengths)))
   print >> outFile, ("MaxSentenceLength\t"+str(max(sentenceLengths)))






## setup the inference algorithm
#from pyro.infer import Trace_ELBO
#svi = SVI(model, guide, optimizer, loss=Trace_ELBO()) #, num_particles=7)
#
#n_steps = 400000
## do gradient steps
#for step in range(1,n_steps):
#    if step % 500 == 1:
#      print "DOING A STEP"
#      print "......."
#      print step
#
#    svi.step(corpus)
#
#    if step % 2000 == 0:
#       print "Saving"
#       save_path = "../../raw-results/"
#       with open(save_path+"/manual_output_funchead_ground_coarse/"+language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
#          print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss","DH_Mean_NoPunct","DH_Sigma_NoPunct", "Distance_Mean_NoPunct", "Distance_Sigma_NoPunct", "Dependency"]))
#          dh_numpy = pyro.get_param_store().get_param("mu_DH").data.numpy()
#          dh_sigma_numpy = pyro.get_param_store().get_param("sigma_DH").data.numpy()
#          dist_numpy = pyro.get_param_store().get_param("mu_Dist").data.numpy()
#          dist_sigma_numpy = pyro.get_param_store().get_param("sigma_Dist").data.numpy()
#
#          for i in range(len(itos_deps)):
#             key = itos_deps[i]
#             dependency = key
#             print >> outFile, "\t".join(map(str,[myID, __file__, counter, crossEntropy, dh_numpy[i], dh_sigma_numpy[i], dist_numpy[i], dist_sigma_numpy[i], dependency]))
#
#
#
#

