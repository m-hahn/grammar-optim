
import random
import sys

objectiveName = "LM"

language = sys.argv[1]
model = sys.argv[2]
BASE_DIR = sys.argv[3]
mode = sys.argv[4]
batchSize = 1
myID = random.randint(0,10000000)

FILENAME = "dotdotdot_COARSE_PLANE_ngram.py"



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
      originalDistanceWeights[key] = (distanceSum[key] / distanceCounts[key])
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable


def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

   # there are the gradients of its children
   if "children_DH" in line:
      for child in line["children_DH"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   result.append(line)
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()]
   line["relevant_logprob_sum"] = allGradients
   if "children_HD" in line:
      for child in line["children_HD"]:
         allGradients = recursivelyLinearize(sentence, child, result, allGradients)
   return allGradients

import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



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
      if line["coarse_dep"].startswith("punct"): # assumes that punctuation does not have non-punctuation dependents!
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
   global mode
   if mode == "none":
      final = linearized
   elif mode == "inwards":
     final = [None for _ in linearized]
     for i in range(len(linearized)):
       if i % 2 == 0:
         target = int(i/2)
       else:
         target = len(linearized)-1-int(i/2)
       assert final[target] is None
       final[target] = linearized[i]
   elif mode == "local_flip":
     final = [None for _ in linearized]
     for i in range(len(linearized)):
       if i % 2 == 0:
         target = i+1
       else:
         target = i-1
       if i % 2 == 0 and i+1 == len(final):
          target = i
       assert final[target] is None
       final[target] = linearized[i]
   elif mode == "interleave":
     final = [None for _ in linearized]
     for i in range(len(linearized)):
       if i < len(linearized)/2:
         target = 2*i
       elif i >= len(linearized)/2:
         target = 2 * (i - len(linearized)/2) + 1
       else:
         assert False,i
       if i+1 == len(final):
           target = i
       assert target < len(final), (i, target, len(linearized), i-len(linearized)/2)
       assert final[target] is None
       final[target] = linearized[i]

   elif mode == "even_odd":
     final = [None for _ in linearized]
     for i in range(len(linearized)):
       if i % 2 == 0:
         target = int(i/2)
       else:
         target = len(linearized)/2+int(i/2)
         if len(linearized) % 2 == 1:
           target += 1
       assert final[target] is None
       final[target] = linearized[i]
   else:
     assert False
   return final, logits


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

   # take from treebank, or randomize
   dhLogits[key] = 2*(random()-0.5)
   dhWeights[i] = dhLogits[key]

   originalDistanceWeights[key] = random()  
   distanceWeights[i] = originalDistanceWeights[key]

import os

if model != "RANDOM" and model != "REAL_REAL":
   temperature = 1.0
   inpModels_path = "../../../../raw-results/"+"/"+BASE_DIR+"/"
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
   for line in data:
      dependency = line[header.index("Dependency")]
      key = dependency
      dhWeights[stoi_deps[key]] = temperature*float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = temperature*float(line[header.index("DistanceWeight")])






words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

assert stoi[itos[5]] == 5


vocab_size = 50000



crossEntropy = 10.0

def encodeWord(w, doTraining):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

def regularisePOS(w, doTraining):
   return w


import torch.cuda
import torch.nn.functional


baselineAverageLoss = 0

counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 
devLossesWords = []
devLossesPOS = []

loss_op = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index = 0)

bigramCounts = {"PLACEHOLDER" : {}}
unigramCountsL = {"PLACEHOLDER" : len(itos)}
unigramCountsR = {"PLACEHOLDER" : len(itos)}

for word in itos:
    unigramCountsL[word] = 1
    unigramCountsR[word] = 1
    bigramCounts[word] = {"PLACEHOLDER" : 1}
    bigramCounts["PLACEHOLDER"][word] = 1

bigramCountsPOSFine = {"PLACEHOLDER" : {}}
unigramCountsPOSFineL = {"PLACEHOLDER" : len(posFine)}
unigramCountsPOSFineR = {"PLACEHOLDER" : len(posFine)}

for word in posFine:
    unigramCountsPOSFineL[word] = 1
    unigramCountsPOSFineR[word] = 1
    bigramCountsPOSFine[word] = {"PLACEHOLDER" : 1}
    bigramCountsPOSFine["PLACEHOLDER"][word] = 1






def doForwardPassTrain(current, train=True):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, current)
       # current is already sorted by length
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"], train) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (regularisePOS(stoi_pos_ptb[x[i-1]["posFine"]]+3, train) if i <= len(x) else 0), batchOrdered))
       posFines = ["SOS"] + [x["posFine"] for x in batchOrdered[0]] + ["EOS"]
       for i in range(len(posFines)-1):
         left = posFines[i]
         right = posFines[i+1]
         if left not in bigramCountsPOSFine:
            bigramCountsPOSFine[left] = {}
         bigramCountsPOSFine[left][right] = bigramCountsPOSFine[left].get(right,0)+1
         unigramCountsPOSFineL[left] = unigramCountsPOSFineL.get(left, 0)+1
         unigramCountsPOSFineR[right] = unigramCountsPOSFineR.get(right, 0)+1
 
       words = ["SOS"] + [x["word"] for x in batchOrdered[0]] + ["EOS"]
       for i in range(len(words)-1):
         left = words[i]
         right = words[i+1]
         if left not in bigramCounts:
            bigramCounts[left] = {}
         bigramCounts[left][right] = bigramCounts[left].get(right,0)+1
         unigramCountsL[left] = unigramCountsL.get(left, 0)+1
         unigramCountsR[right] = unigramCountsR.get(right, 0)+1
 
from math import log

def doForwardPassEvaluate(current, train=True):
       global counter
       global crossEntropy
       global printHere
       global devLosses
       global baselineAverageLoss
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, current)
       # current is already sorted by length
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"], train) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (regularisePOS(stoi_pos_ptb[x[i-1]["posFine"]]+3, train) if i <= len(x) else 0), batchOrdered))



       posFines = ["SOS"] + [x["posFine"] for x in batchOrdered[0]] + ["EOS"]
       surprisalPOS = 0
       delta = 0.5
       for i in range(len(posFines)-1):
         left = posFines[i]
         right = posFines[i+1]
         bigramCountsPOSFineLeft = bigramCountsPOSFine.get(left, {})
         bigramCount = bigramCountsPOSFineLeft.get(right, 0)


         unigramCountsPOSFineLLeft = unigramCountsPOSFineL.get(left, 0)
         unigramCountsPOSFineRRight = unigramCountsPOSFineR.get(right, 0)
         prob = (max(bigramCount-delta, 0.0) + float(unigramCountsPOSFineRRight)/totalUnigramCount * delta * len(bigramCountsPOSFineLeft))/unigramCountsPOSFineLLeft
         assert prob <= 1.0
         surprisalPOS -= log(prob)
         if printHere and i > 0 and i < len(batchOrdered[0]):
             print "\t".join(map(str,[batchOrdered[0][i]["posFine"], batchOrdered[0][i]["posFine"], log(prob), log(float(unigramCountsPOSFineRRight)/totalUnigramCount)]))



       words = ["SOS"] + [x["word"] for x in batchOrdered[0]] + ["EOS"]
       surprisalWord = 0
       delta = 0.5
       for i in range(len(words)-1):
         left = words[i]
         right = words[i+1]
         bigramCountsLeft = bigramCounts.get(left, {})
         bigramCount = bigramCountsLeft.get(right, 0)


         unigramCountsLLeft = unigramCountsL.get(left, 0)
         unigramCountsRRight = unigramCountsR.get(right, 0)
         prob = (max(bigramCount-delta, 0.0) + float(unigramCountsRRight)/totalUnigramCount * delta * len(bigramCountsLeft))/unigramCountsLLeft
         assert prob <= 1.0
         surprisalWord -= log(prob)
         if printHere and i > 0 and i < len(batchOrdered[0]):
             print "\t".join(map(str,[batchOrdered[0][i]["word"], batchOrdered[0][i]["posFine"], log(prob), log(float(unigramCountsRRight)/totalUnigramCount)]))
       _ = 0
       return _, _, _, surprisalWord+surprisalPOS, len(words)+1,surprisalWord, surprisalPOS


def computeDevLoss():
   global printHere
   global counter
   devLoss = 0.0
   devLossWords = 0.0
   devLossPOS = 0.0
   devWords = 0
   corpusDev = CorpusIteratorFuncHead(language,"dev").iterator(rejectShortSentences = True)

   while True:
     try:
        batch = map(lambda x:next(corpusDev), 10*range(batchSize))
     except StopIteration:
        break
     batch = sorted(batch, key=len)
     partitions = range(10)
     shuffle(partitions)
     for partition in partitions:
        counter += 1
        printHere = (counter % 50 == 0)
        current = batch[partition*batchSize:(partition+1)*batchSize]
 
        _, _, _, newLoss, newWords, lossWords, lossPOS = doForwardPassEvaluate(current, train=False)

        devLoss += newLoss
        devWords += newWords
        devLossWords += lossWords
        devLossPOS += lossPOS
   return devLoss/devWords, devLossWords/devWords, devLossPOS/devWords

if True:
  corpus = CorpusIteratorFuncHead(language).iterator(rejectShortSentences = True)


  while True:
    try:
       batch = map(lambda x:next(corpus), 10*range(batchSize))
    except StopIteration:
       break
    batch = sorted(batch, key=len)
    partitions = range(10)
    shuffle(partitions)
    for partition in partitions:
       counter += 1
       printHere = (counter % 100 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       doForwardPassTrain(current)
  print(bigramCounts)

  totalUnigramCount = sum([y for x,y in unigramCountsR.iteritems()])

  if True: #counter % 10000 == 0:
          newDevLoss, newDevLossWords, newDevLossPOS = computeDevLoss()
          devLosses.append(newDevLoss)
          devLossesWords.append(newDevLossWords)
          devLossesPOS.append(newDevLossPOS)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          print "Saving"
          save_path = "../../../../raw-results/"
          if True:
             print(save_path+"/language_modeling_adversarial/"+language+"_"+FILENAME+"_languageModel_performance_"+model+"_"+str(myID)+".tsv")
             with open(save_path+"/language_modeling_adversarial/"+language+"_"+FILENAME+"_languageModel_performance_"+model+"_"+str(myID)+".tsv", "w") as outFile:
                print >> outFile, "\t".join(sys.argv)
                print >> outFile, "\t".join(map(str, devLosses))
                print >> outFile, "\t".join(map(str, devLossesWords))
                print >> outFile, "\t".join(map(str, devLossesPOS))




          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             print devLossesWords
             print devLossesPOS
             quit()

