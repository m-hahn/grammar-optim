#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys



language = sys.argv[1]
languageCode = sys.argv[2]
model = sys.argv[3]
temperature = sys.argv[4] #) if len(sys.argv) > 4 else 1.0
BASE_DIR = sys.argv[5]

assert temperature == "Infinity"

batchSize = 1




posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp, sqrt
from random import random, shuffle, randint
import os

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

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
#torch.exp(line["ordering_decision_log_probability"]).data.numpy(),
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
#      sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



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
      if key not in stoi_deps:
          continue
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

assert stoi[itos[5]] == 5

#print dhLogits

#for sentence in getNextSentence():
#   print orderSentence(sentence, dhLogits)

vocab_size = 50000


# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 50).cuda()
#pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
#pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()


#baseline = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim=1).cuda()
#baseline_upos = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim=1).cuda()
#baseline_ppos = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=1).cuda()





crossEntropy = 10.0

def encodeWord(w, doTraining):
#   if doTraining and random() < input_noise and len(stoi) > 10:
 #     return 3+randint(0, len(itos)-1)
   return stoi[w]+3 if stoi[w] < vocab_size else 1

def regularisePOS(w, doTraining):
   return w
#   if doTraining and random() < 0.01 and len(stoi_pos_ptb) > 10:
#      return 3+randint(0, len(stoi_pos_ptb)-1)
#   return w

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



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

         #print(left, bigramCountsPOSFineLeft, bigramCount)

         unigramCountsPOSFineLLeft = unigramCountsPOSFineL.get(left, 0)
         unigramCountsPOSFineRRight = unigramCountsPOSFineR.get(right, 0)
         prob = (max(bigramCount-delta, 0.0) + float(unigramCountsPOSFineRRight)/totalUnigramCount * delta * len(bigramCountsPOSFineLeft))/unigramCountsPOSFineLLeft
#         totalProb = 0
#         for posFine in itos + ["PLACEHOLDER"]:
#             bigramCount1 = bigramCountsPOSFineLeft.get(posFine, 0)
#             unigramCountsPOSFineRRight1 = unigramCountsPOSFineR.get(posFine, 0)
#             prob2 = (max(bigramCount1-delta, 0.0) + float(unigramCountsPOSFineRRight1)/totalUnigramCount * delta * len(bigramCountsPOSFineLeft))/unigramCountsPOSFineLLeft
#             totalProb += prob2
#         print(totalProb)
#         assert totalProb <= 1.01, (totalProb, left)
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

         #print(left, bigramCountsLeft, bigramCount)

         unigramCountsLLeft = unigramCountsL.get(left, 0)
         unigramCountsRRight = unigramCountsR.get(right, 0)
         prob = (max(bigramCount-delta, 0.0) + float(unigramCountsRRight)/totalUnigramCount * delta * len(bigramCountsLeft))/unigramCountsLLeft
#         totalProb = 0
#         for word in itos + ["PLACEHOLDER"]:
#             bigramCount1 = bigramCountsLeft.get(word, 0)
#             unigramCountsRRight1 = unigramCountsR.get(word, 0)
#             prob2 = (max(bigramCount1-delta, 0.0) + float(unigramCountsRRight1)/totalUnigramCount * delta * len(bigramCountsLeft))/unigramCountsLLeft
#             totalProb += prob2
#         print(totalProb)
#         assert totalProb <= 1.01, (totalProb, left)
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
#   corpusDev = getNextSentence("dev")
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
        printHere = (counter % 500 == 0)
        current = batch[partition*batchSize:(partition+1)*batchSize]
 
        _, _, _, newLoss, newWords, lossWords, lossPOS = doForwardPassEvaluate(current, train=False)

        devLoss += newLoss
        devWords += newWords
        devLossWords += lossWords
        devLossPOS += lossPOS
   return devLoss/devWords, devLossWords/devWords, devLossPOS/devWords

if True:
#  corpus = getNextSentence("train")
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
       printHere = (counter % 500 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       doForwardPassTrain(current)
  #print(bigramCounts)

  totalUnigramCount = sum([y for x,y in unigramCountsR.iteritems()])

  if True: #counter % 10000 == 0:
          newDevLoss, newDevLossWords, newDevLossPOS = computeDevLoss()
          devLosses.append(newDevLoss)
          devLossesWords.append(newDevLossWords)
          devLossesPOS.append(newDevLossPOS)
          print(devLossesWords)

if True:
          print "Saving"
          save_path = "/juicier/scr120/scr/mhahn/deps/"
          with open(save_path+"/info_locality/"+language+"_"+__file__.split("/")[-1]+"_model_"+model+"_SHORT.tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,[language, model,BASE_DIR, newDevLossWords]))


