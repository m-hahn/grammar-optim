#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

from collections import deque

#objectiveName = "gradients_WordSurp"

language = sys.argv[1]
languageCode = sys.argv[2]
model = sys.argv[3]
temperature = sys.argv[4] #) if len(sys.argv) > 4 else 1.0
BASE_DIR = sys.argv[5]
if len(sys.argv) > 6:
   printDetailedData = (sys.argv[6] == "True")
else:
   printDetailedData = False



assert model == "REAL_REAL"
assert BASE_DIR == "REAL_REAL"

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
#       childrenLinearized = []
#       while len(remainingChildren) > 0:
       logits = [(x, distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]]) for x in remainingChildren]
       logits = sorted(logits, key=lambda x:x[1], reverse=(not reverseSoftmax))
       childrenLinearized = map(lambda x:x[0], logits)
           
#           #print logits
#           if reverseSoftmax:
#              
#              logits = -logits
#           #print (reverseSoftmax, logits)
#           softmax = softmax_layer(logits.view(1,-1)).view(-1)
#           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
#    #       log_probability = torch.log(softmax[selected])
#   #        assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
#  #         sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
#           childrenLinearized.append(remainingChildren[selected])
#           del remainingChildren[selected]
       return childrenLinearized           
#           softmax = torch.distributions.Categorical(logits=logits)
#           selected = softmax.sample()
#           print selected
#           quit()
#           softmax = torch.cat(logits)



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
#      probability = 1/(1 + torch.exp(-dhLogit))
      dhSampled = (dhLogit > 0) #(random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      #line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy()[0],
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      #sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



   if model != "REAL_REAL":
      assert False
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


#         shuffle(line["children_HD"])
   
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



      #originalLoss = float(line[header.index("AverageLoss")])

#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]
#


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


## 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 50).cuda()
##pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
##pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()
#
#
#baseline = nn.Linear(50, 1).cuda()
#
#dropout = nn.Dropout(0.5).cuda()
#
#rnn = nn.LSTM(70, 128, 2).cuda()
#for name, param in rnn.named_parameters():
#  if 'bias' in name:
#     nn.init.constant(param, 0.0)
#  elif 'weight' in name:
#     nn.init.xavier_normal(param)
#
#decoder = nn.Linear(128,vocab_size+3).cuda()
##pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()
#
#components = [word_embeddings, rnn, decoder,  baseline] # pos_ptb_decoder,
## pos_u_embeddings, pos_p_embeddings, 
#
#def parameters():
# for c in components:
#   for param in c.parameters():
#      yield param
## yield dhWeights
## yield distanceWeights
#
##for pa in parameters():
##  print pa
#
#initrange = 0.1
#word_embeddings.weight.data.uniform_(-initrange, initrange)
##pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
##pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#decoder.bias.data.fill_(0)
#decoder.weight.data.uniform_(-initrange, initrange)
##pos_ptb_decoder.bias.data.fill_(0)
##pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)
#baseline.bias.data.fill_(0)
#baseline.weight.data.uniform_(-initrange, initrange)

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
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0), batchOrdered))

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       #for c in components:
       #   c.zero_grad()

       #momentum = 0.0
       #assert momentum == 0.0 # her taking offline estimate, so no momentum
       #for p in  [dhWeights, distanceWeights]:
       #   if p.grad is not None:
       #      p.grad.data = p.grad.data.mul(momentum)


       totalQuality = 0.0

# (Variable(weight.new(2, bsz, self.nhid).zero_()),Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
#       for i in range(maxLength+1):
       if True:
           # TODO word dropout could also be added: randomly sprinkle `1' (UNK) in the LongTensor (not into input_words -- this will also be used for the softmax!)
 #          words_layer = word_embeddings(Variable(torch.LongTensor(input_words)).cuda())
 #          #pos_u_layer = pos_u_embeddings(Variable(torch.LongTensor(input_pos_u)).cuda())
 #          #pos_p_layer = pos_p_embeddings(Variable(torch.LongTensor(input_pos_p)).cuda())
 #          inputEmbeddings = dropout(words_layer) #torch.cat([words_layer, pos_u_layer, pos_p_layer], dim=2))
##           print hidden
 #          output, hidden = rnn(inputEmbeddings, hidden)
##           print maxLength
 #          
##           print inputEmbeddings
##           print output
 #          baseline_predictions = baseline(words_layer.detach())

 #          # word logits
 #          word_logits = decoder(dropout(output))
 #          word_logits = word_logits.view(-1, vocab_size+3)
 #          word_softmax = logsoftmax(word_logits)
 #          word_softmax = word_softmax.view(-1, batchSize, vocab_size+3)

 #          # pos logits
##           pos_logits = pos_ptb_decoder(dropout(output))
##           pos_logits = pos_logits.view(-1, len(posFine)+3)
##           pos_softmax = logsoftmax(pos_logits)
 ##          pos_softmax = pos_softmax.view(-1, batchSize, len(posFine)+3)

 #       
 #
##           print word_logits
##           print predictions
 #          lossesWord = [[None]*batchSize for i in range(maxLength+1)]
#           lossesPOS = [[None]*batchSize for i in range(maxLength+1)]

#           print word_logits
           totalDepLength = 0
           byType = []
           for i in range(1,len(input_words)): #range(1,maxLength+1): # don't include i==0
              for j in range(batchSize):
                 if input_words[i][j] != 0:
                    if batchOrdered[j][i-1]["head"] == 0:
                       realHead = 0
                    else:
                       realHead = batchOrdered[j][i-1]["reordered_head"] # this starts at 1, so just right for the purposes here
                    if batchOrdered[j][i-1]["coarse_dep"] == "root":
#                       print batchOrdered[j][i-1]
                       continue
                    # discard punctuation
#                    if batchOrdered[j][i-1]["dep"] == ".":
#                       continue
                    # to make sure reward attribution considers this correctly
#                    registerAt = max(i, realHead)
#                    print (i, realHead)
                    depLength = abs(i - realHead) # - 1 # to be consistent with Richard's stuff
                    assert depLength >= 1
                    totalDepLength += depLength
                    byType.append((batchOrdered[j][i-1]["coarse_dep"], depLength))
#                    if j == 0:
 #                     print ["DECISION_PROB",batchOrdered[j][i]["relevant_logprob_sum"].data.numpy()[0] ]
                    if input_words[i] > 2 and j == 0 and printHere:
                       print [itos[input_words[i][j]-3], itos_pos_ptb[input_pos_p[i][j]-3] ]
                    wordNum += 1

#       if printHere:
#         print totalDepLength/wordNum
##           losses = loss(predictions, input_words[i+1]) 
##           print losses
##    for i, sentence in enumerate(batchOrderLogits):
##       embeddingsLayer
#         print ["CROSS ENTROPY", crossEntropy]
       if wordNum > 0:
          crossEntropy = 0.99 * crossEntropy + 0.01 * (totalDepLength/wordNum)
       else:
          assert totalDepLength == 0
       numberOfWords = wordNum
       return (totalDepLength, numberOfWords, byType)


#def  doBackwardPass(loss, baselineLoss, policy_related_loss):
#       global lastDevLoss
#       global failedDevRuns
#       if printHere:
#         print "BACKWARD 1"
#       policy_related_loss.backward()
#       if printHere:
#         print "BACKWARD 2"
#
##       loss += entropy_weight * neg_entropy
##       loss += lr_policy * policyGradientLoss
#
#       loss += baselineLoss # lives on GPU
#       loss.backward()
#       if printHere:
#         print "BACKWARD 3 "+my_fileName+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)
#         print devLosses
#       torch.nn.utils.clip_grad_norm(parameters(), 5.0, norm_type='inf')
#       for param in parameters():
#         #print "UPDATING"
#         if param.grad is None:
#           print "WARNING: None gradient"
#           continue
#         param.data.sub_(lr_lm * param.grad.data)
#        
#       dhGradients_WSurp.append(dhWeights.grad.data.numpy())
#       distanceGradients_WSurp.append(distanceWeights.grad.data.numpy())




#def computeDevLoss():
#   global printHere
#   global counter
#   devLoss = 0.0
#   devWords = 0
#   corpusDev = CorpusIteratorFuncHead(language,"dev").iterator()
#   while True:
#     try:
#        batch = map(lambda x:next(corpusDev), 10*range(batchSize))
#     except StopIteration:
#        break
#     batch = sorted(batch, key=len)
#     partitions = range(10)
#     shuffle(partitions)
#     for partition in partitions:
#        counter += 1
#        printHere = (counter % 5 == 0)
#        current = batch[partition*batchSize:(partition+1)*batchSize]
# 
#        _, _, _, newLoss, newWords = doForwardPass(current)
#        devLoss += newLoss
#        devWords += newWords
#   return devLoss/devWords

#dhGradients_WSurp = deque(maxlen=50000) # * corpus.length())
#distanceGradients_WSurp = deque(maxlen=50000) # * corpus.length())

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
       batch = map(lambda x:next(corpusIterator), 10*range(batchSize))
    except StopIteration:
       break
    batch = sorted(batch, key=len)
    partitions = range(10)
    shuffle(partitions)
    
    for partition in partitions:
       counter += 1
       printHere = (counter % 100 == 0)
       current = batch[partition*batchSize:(partition+1)*batchSize]

       depLength = doForwardPass(current)
       depLengths.append(depLength)
       if counter % 100 == 0:
          print  "Average dep length "+str(sum(map(lambda x:x[0], depLengths))/float(len(depLengths)))+" "+str(counter)+" "+str(float(counter)/corpus.length())+"%"
#       doBackwardPass(loss, baselineLoss, policy_related_loss)


#       if counter % 20 == 0:
#          dhGradients_WSurp_mean = sum([x for x in dhGradients_WSurp])/len(dhGradients_WSurp) # deque(maxlen=corpus.length())
#          distanceGradients_WSurp_mean = sum([x for x in distanceGradients_WSurp])/len(distanceGradients_WSurp)
#
#
#          print "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss","Head","D_WordSurp_DH_Weight","Dependency","Dependent","D_WordSurp_DistanceWeight", "ObjectiveName"]))
#          for i in range(len(itos_deps)):
#             key = itos_deps[i]
#             dhWeight = dhGradients_WSurp_mean[i]
#             distanceWeight = distanceGradients_WSurp_mean[i]
#             head, dependency, dependent = key
#             print "\t".join(map(str,[myID, my_fileName, counter, crossEntropy, head, dhWeight, dependency, dependent, distanceWeight, entropy_weight, objectiveName]))




#       if counter % 10000 == 0:
#          newDevLoss = computeDevLoss()
#          devLosses.append(newDevLoss)
#          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
#          if lastDevLoss is None or newDevLoss < lastDevLoss:
#             lastDevLoss = newDevLoss
#             failedDevRuns = 0
#          else:
#             failedDevRuns += 1
#             print "Skip saving, hoping for better model"
#             lr_lm *= 0.5
#             continue

originalCounter = 100000
if True:
          print "Saving"
          save_path = "../../raw-results/"

          with open(save_path+"/dependency_length/summaries/"+language+"_"+my_fileName+"_model_"+str(myID)+"_"+model+"_SHORT.tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,["Language", "FileName","ModelName","Counter", "Model", "Temperature", "OriginalCounter", "AverageLengthPerWord", "AverageLengthPerSentence", "OriginalLoss"]))
             lengths = []
             perSentenceLengths = []
             for y, l in enumerate(depLengths):
               perSentenceLengths.append(0)
               for e in l[2]:
                 lengths.append(e[1])
                 perSentenceLengths[-1] += e[1]
             print >> outFile, "\t".join(map(str,[language, myID, my_fileName, counter, model, temperature, originalCounter,float(sum(lengths))/len(lengths), float(sum(perSentenceLengths))/len(perSentenceLengths), "NA"]))

          if printDetailedData:
            with open(save_path+"/dependency_length/"+language+"_"+my_fileName+"_model_"+model+".tsv", "w") as outFile:
               print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "Model", "SentenceNumber", "Type", "Length", "Temperature"]))
               for y, l in enumerate(depLengths):
                 for e in l[2]:
                   print >> outFile, "\t".join(map(str,[myID, my_fileName, counter, model, y, e[0], e[1], temperature]))
  


#          print "Stopping after 10000 sentences"
          quit()

#dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
#for i, key in enumerate(itos_deps):
#
#   # take from treebank, or randomize
#   dhLogits[key] = 2*(random()-0.5)
#   dhWeights.data[i] = dhLogits[key]
#
#   originalDistanceWeights[key] = random()  
#   distanceWeights.data[i] = originalDistanceWeights[key]
#
#
#
