#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

from collections import deque


language = sys.argv[1]
languageCode = sys.argv[2]
model = sys.argv[3]
temperature = sys.argv[4] #) if len(sys.argv) > 4 else 1.0
BASE_DIR = sys.argv[5]
if len(sys.argv) > 6:
   printDetailedData = (sys.argv[6] == "True")
else:
   printDetailedData = False

assert temperature == "Infinity"

myID = random.randint(0,10000000)

posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp
from random import random, shuffle
import os

conll_header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

my_fileName = __file__.split("/")[-1]

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

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum # + sum(line.get("children_decisions_logprobs",[]))
  # if "linearization_logprobability" in line:
  #    allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at this word, but nothing to the left of it
  # else:
  #    assert line["dep"] == "root"

   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()[0]]
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
       return childrenLinearized           



def orderSentence(sentence, dhLogits, printThings):
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
      dhLogit = dhWeights[stoi_deps[key]]
      dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())

      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])



   for line in sentence:
      if "children_DH" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_DH"][:], False)
         line["children_DH"] = childrenLinearized
      if "children_HD" in line:
         childrenLinearized = orderChildrenRelative(sentence, line["children_HD"][:], True)
         line["children_HD"] = childrenLinearized

   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
   if printThings:
     print " ".join(map(lambda x:x["word"], sentence))
     print " ".join(map(lambda x:x["word"], linearized))


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
#      print x
      moved[x["index"]-1] = i
 #  print moved
   for i,x in enumerate(linearized):
  #    print x
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



if model != "RANDOM":
   inpModels_path = "/juicier/scr120/scr/mhahn/deps/"+BASE_DIR+"/"
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
    
   #print header
   #quit()
   # there might be a divergence because 'inferWeights...' models did not necessarily run on the full set of corpora per language (if there is no AllCorpora in the filename)
   #assert len(data) == len(itos_deps), [len(data), len(itos_deps)]
   if "CoarseDependency" not in header:
     header[header.index("Dependency")] = "CoarseDependency"
   if "DH_Weight" not in header:
     header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
     header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
   #   print line
#      head = line[header.index("Head")]
 #     dependent = line[header.index("Dependent")]
      dependency = line[header.index("CoarseDependency")]
      key = dependency
      dhWeights[stoi_deps[key]] = float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = float(line[header.index("DistanceWeight")])

      if "Counter" in header:
        originalCounter = int(line[header.index("Counter")])
      else:
        originalCounter = 200000
      #originalLoss = float(line[header.index("AverageLoss")])


words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))
#print stoi
#print itos[5]
#print stoi[itos[5]]

if len(itos) > 6:
   assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size = 50000



batchSize = 1

lr_lm = 0.1


crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional




counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 

pairsCounts = {}
targetCounts = {}
sourceCount = 0

def collectPairsCounts(source, target):
    if source not in pairsCounts:
       pairsCounts[source] = {"_TOTAL_":0, target : 0}
    if target not in pairsCounts[source]:
       pairsCounts[source][target] = 0
    pairsCounts[source]["_TOTAL_"] +=1
    pairsCounts[source][target] +=1
    targetCounts[target] = targetCounts.get(target,0) + 1
    global sourceCount
    sourceCount += 1


def doForwardPass(current):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, current)
       # current is already sorted by length
       maxLength = lengths[int(0.8*batchSize)]
       input_words = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       totalQuality = 0.0
       if True:
           totalDepLength = 0
           byType = []
           for i in range(1,len(input_words)): #range(1,maxLength+1): # don't include i==0
              for j in range(batchSize):
                 source = input_words[i-1][j]
                 target = input_words[i][j]
                 collectPairsCounts(source, target)



assert batchSize == 1

depLengths = []
#while True:
if True:
  corpus = CorpusIteratorFuncHead(language,"train")
  corpusIterator = corpus.iterator()
  if corpus.length() == 0:
     quit()
  while True:
    try:
       batch = map(lambda x:next(corpusIterator), 1*range(batchSize))
    except StopIteration:
       break
    batch = sorted(batch, key=len)
    partitions = range(1)
    shuffle(partitions)
    
    for partition in partitions:
       counter += 1
       printHere = (counter % 500 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       depLength = doForwardPass(current)
       depLengths.append(depLength)



#print(pairsCounts)
print(sourceCount)
#print(targetCounts)
mi = 0
for source in pairsCounts:
  sumTarget = 0
  pSource = float(pairsCounts[source]["_TOTAL_"]) / sourceCount
  for target, count in pairsCounts[source].iteritems():
     if target == "_TOTAL_":
       continue
     pTargetSource = float(count) / pairsCounts[source]["_TOTAL_"]
     pTarget = float(targetCounts[target]) / sourceCount
     mi += pSource * (pTargetSource * (log(pTargetSource) - log(pTarget)))
     sumTarget += pTargetSource
print(mi)





if True:
          print "Saving"
          save_path = "/juicier/scr120/scr/mhahn/deps/"
          with open(save_path+"/info_locality/"+language+"_"+my_fileName+"_model_"+model+"_SHORT.tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,[language, model,BASE_DIR, mi]))





