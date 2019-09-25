



import random
import sys

objectiveName = "LM"

language = sys.argv[1]
languageCode = sys.argv[2]
entropy_weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
lr_policy = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001
momentum = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
lr_baseline = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0 # this will be multiplied with lr_lm
dropout_prob = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
lr_lm = float(sys.argv[8]) if len(sys.argv) > 8 else 0.1
batchSize = int(sys.argv[9]) if len(sys.argv) > 9 else 1
model = sys.argv[10]
BASE_DIR = sys.argv[11]

assert "pureUD" in BASE_DIR

myID = random.randint(0,10000000)

FILENAME = __file__

#with open("../../../raw-results/LOG"+language+"_"+FILENAME+"_model_"+str(myID)+".txt", "w") as outFile:
#    print >> outFile, " ".join(sys.argv)


posUni = set()
posFine = set()



from math import log, exp, sqrt
from random import random, shuffle, randint, choice
import os


from corpusIterator import CorpusIterator

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
     for sentence in CorpusIterator(language,partition).iterator():
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
   allGradients = gradients_from_the_left_sum

   # there are the gradients of its children
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

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]] for x in remainingChildren]
           if reverseSoftmax:
              logits = [-x for x in logits]
           softmax = logits
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
      line["ordering_decision_log_probability"] = 0 

      
     
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

import os

if model not in ["RANDOM_pureUD", "RANDOM_pureUD2", "RANDOM_pureUD3", "RANDOM_pureUD4", "RANDOM_pureUD5"] and model != "REAL_REAL" and model != "RLR":
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
   if "DH_Weight" not in header:
      header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
      header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
      dependency = line[header.index("Dependency")]
      key = dependency
      dhWeights[stoi_deps[key]] = temperature*float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = temperature*float(line[header.index("DistanceWeight")])
elif model == "RANDOM_pureUD":
   assert BASE_DIR == "RANDOM_pureUD"
   save_path = "../../../../raw-results/"
   with open(save_path+"/manual_output_funchead_RANDOM_pureUD/"+language+"_"+"RANDOM_pureUD"+"_model_"+str(myID)+".tsv", "w") as outFile:
      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
      for i in range(len(itos_deps)):
         key = itos_deps[i]
         dhWeight = dhWeights[i]
         distanceWeight = distanceWeights[i]
         dependency = key
         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))
elif model == "RANDOM_pureUD2":
   assert BASE_DIR == "RANDOM_pureUD2"
   save_path = "/u/scr/mhahn/deps/"
   #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
   with open(save_path+"/manual_output_funchead_RANDOM_pureUD2/"+language+"_"+"RANDOM_pureUD2"+"_model_"+str(myID)+".tsv", "w") as outFile:
      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
      for i in range(len(itos_deps)):
         key = itos_deps[i]
         dhWeight = dhWeights[i]
         distanceWeight = distanceWeights[i]
         dependency = key
         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))
elif model == "RANDOM_pureUD3":
   assert BASE_DIR == "RANDOM_pureUD3"
   save_path = "/u/scr/mhahn/deps/"
   #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
   with open(save_path+"/manual_output_funchead_RANDOM_pureUD3/"+language+"_"+"RANDOM_pureUD3"+"_model_"+str(myID)+".tsv", "w") as outFile:
      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
      for i in range(len(itos_deps)):
         key = itos_deps[i]
         dhWeight = dhWeights[i]
         distanceWeight = distanceWeights[i]
         dependency = key
         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))
elif model == "RANDOM_pureUD4":
   assert BASE_DIR == "RANDOM_pureUD4"
   save_path = "/u/scr/mhahn/deps/"
   #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
   with open(save_path+"/manual_output_funchead_RANDOM_pureUD4/"+language+"_"+"RANDOM_pureUD4"+"_model_"+str(myID)+".tsv", "w") as outFile:
      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
      for i in range(len(itos_deps)):
         key = itos_deps[i]
         dhWeight = dhWeights[i]
         distanceWeight = distanceWeights[i]
         dependency = key
         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))
elif model == "RANDOM_pureUD5":
   assert BASE_DIR == "RANDOM_pureUD5"
   save_path = "/u/scr/mhahn/deps/"
   #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
   with open(save_path+"/manual_output_funchead_RANDOM_pureUD5/"+language+"_"+"RANDOM_pureUD5"+"_model_"+str(myID)+".tsv", "w") as outFile:
      print >> outFile, "\t".join(map(str,["FileName","DH_Weight", "CoarseDependency","DistanceWeight" ]))
      for i in range(len(itos_deps)):
         key = itos_deps[i]
         dhWeight = dhWeights[i]
         distanceWeight = distanceWeights[i]
         dependency = key
         print >> outFile, "\t".join(map(str,[myID, dhWeight, dependency, distanceWeight]))




words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

assert stoi[itos[5]] == 5


vocab_size = 50000


word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = 50).cuda()
pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()



dropout = nn.Dropout(dropout_prob).cuda()

rnn = nn.LSTM(70, 128, 2).cuda()
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(128,vocab_size+3).cuda()
pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()

components = [word_embeddings, pos_u_embeddings, pos_p_embeddings, rnn, decoder, pos_ptb_decoder ]


def parameters_lm():
 for c in components:
   for param in c.parameters():
      yield param



def parameters():
 for c in components:
   for param in c.parameters():
      yield param

parameters_cached = [x for x in parameters()]

parameters_lm_cached = [x for x in parameters_lm()]


initrange = 0.1
word_embeddings.weight.data.uniform_(-initrange, initrange)
pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)
pos_ptb_decoder.bias.data.fill_(0)
pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)



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



def doForwardPass(current, train=True):
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

       hidden = None 
       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       for c in components:
          c.zero_grad()


       totalQuality = 0.0

       if True:
           wordIndices = Variable(torch.LongTensor(input_words)).cuda()
           pos_p_indices = Variable(torch.LongTensor(input_pos_p)).cuda()
           words_layer = word_embeddings(wordIndices)
           pos_u_indices = Variable(torch.LongTensor(input_pos_u)).cuda()
           pos_u_layer = pos_u_embeddings(pos_u_indices)
           pos_p_layer = pos_p_embeddings(pos_p_indices)
           input_concat = torch.cat([words_layer, pos_u_layer, pos_p_layer], dim=2)
           inputEmbeddings = dropout(input_concat) if train else input_concat



           output, hidden = rnn(inputEmbeddings, hidden)

           droppedOutput = dropout(output) if train else output

           # word logits
           word_logits = decoder(droppedOutput)
           word_logits = word_logits.view(-1, vocab_size+3)
           word_softmax = logsoftmax(word_logits)
           word_softmax = word_softmax.view(-1, batchSize, vocab_size+3)

           # pos logits
           pos_logits = pos_ptb_decoder(droppedOutput)
           pos_logits = pos_logits.view(-1, len(posFine)+3)
           pos_softmax = logsoftmax(pos_logits)
           pos_softmax = pos_softmax.view(-1, batchSize, len(posFine)+3)

           lossesWord = [[None]*batchSize for i in range(maxLength+1)]
           lossesPOS = [[None]*batchSize for i in range(maxLength+1)]

           lossesWord_tensor = loss_op(word_softmax.view(-1, vocab_size+3)[:-1], wordIndices[1:].view(-1)).view(-1, batchSize)
           lossesPOS_tensor = loss_op(pos_softmax.view(-1, len(posFine)+3)[:-1], pos_p_indices[1:].view(-1)).view(-1, batchSize)

           reward = (lossesWord_tensor + lossesPOS_tensor).detach()

           baselineLoss = 0
 
           for i in range(0,len(input_words)-1): 
              for j in range(batchSize):
                 if input_words[i+1][j] != 0:
                    if input_words[i+1] > 2 and j == 0 and printHere:
                       print [itos[input_words[i+1][j]-3], itos_pos_ptb[input_pos_p[i+1][j]-3], lossesWord_tensor[i][j].data.cpu().numpy(), lossesPOS_tensor[i][j].data.cpu().numpy()]
                    wordNum += 1

       lossWords = lossesWord_tensor.sum()
       lossPOS = lossesPOS_tensor.sum()
       loss += lossWords
       loss += lossPOS
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
         print baselineAverageLoss
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum

       return loss, baselineLoss, 0, totalQuality, numberOfWords, lossWords.data.cpu().numpy(), lossPOS.data.cpu().numpy()


def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns
       if printHere:
         print "BACKWARD 1"

       if printHere:
         print "BACKWARD 2"

       loss.backward()
       if printHere:
         print sys.argv
         print "BACKWARD 3 "+FILENAME+" "+language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+(" ".join(map(str,["ENTROPY", entropy_weight, "LR_POLICY", lr_policy, "MOMENTUM", momentum])))
         print devLosses
         print devLossesWords
         print devLossesPOS

       counterHere = 0
       for param in parameters_cached:
         counterHere += 1
         #print "UPDATING"
         if counter < 50 and (param is distanceWeights or param is dhWeights): # allow baseline to warum up
             continue
         if param.grad is None:
           print counterHere
           print "WARNING: None gradient"
           continue
         param.data.sub_(lr_lm * param.grad.data)


def computeDevLoss():
   global printHere
   global counter
   devLoss = 0.0
   devLossWords = 0.0
   devLossPOS = 0.0
   devWords = 0
   corpusDev = CorpusIterator(language,"dev").iterator(rejectShortSentences = True)

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
 
        _, _, _, newLoss, newWords, lossWords, lossPOS = doForwardPass(current, train=False)
        devLoss += newLoss
        devWords += newWords
        devLossWords += lossWords
        devLossPOS += lossPOS
   return devLoss/devWords, devLossWords/devWords, devLossPOS/devWords

while True:
  corpus = CorpusIterator(language).iterator(rejectShortSentences = True)


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

       loss, baselineLoss, policy_related_loss, _, wordNumInPass, lossWords, lossPOS = doForwardPass(current)
       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"

  # Run on held-out set
  if True: 
          newDevLoss, newDevLossWords, newDevLossPOS = computeDevLoss()
          devLosses.append(newDevLoss)
          devLossesWords.append(newDevLossWords)
          devLossesPOS.append(newDevLossPOS)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
          print "Saving"
          save_path = "../../../../raw-results/"
          with open(save_path+"/language_modeling_coarse_plane_fixed_pureUD/"+language+"_"+FILENAME+"_languageModel_performance_"+model+"_"+str(myID)+".tsv", "w") as outFile:
             print >> outFile, language
             print >> outFile, "\t".join(map(str, devLosses))
             print >> outFile, "\t".join(map(str, devLossesWords))
             print >> outFile, "\t".join(map(str, devLossesPOS))
             print >> outFile, "\t".join(map(str, sys.argv))




          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             quit()

