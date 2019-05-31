#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

objectiveName = "graphParser"

print sys.argv

language = sys.argv[1]
prescribedID = sys.argv[2]
lr_policy = float(sys.argv[3])
assert lr_policy < 1.0
momentum_policy = float(sys.argv[4])
assert momentum_policy < 1.0
entropy_weight = float(sys.argv[5])
assert entropy_weight >= 0
lr_lm = float(sys.argv[6]) #random.choice([0.01, 0.002, 0.001, 0.0005, 0.0002, 0.0001])
assert lr_lm < 0.1
beta1 = float(sys.argv[7]) #0.9
assert beta1 == 0.9
beta2 = float(sys.argv[8]) #random.choice([0.9, 0.999])
assert beta2 in [0.9, 0.999]
dropout_rate = float(sys.argv[9]) #random.choice([0.0, 0.1, 0.2, 0.3])
assert dropout_rate in [0.0, 0.1, 0.2, 0.3]
tokensPerBatch = int(sys.argv[10]) # random.choice([50, 100, 200, 500, 1000, 2000])
assert tokensPerBatch in [10, 20, 30, 50, 100, 200, 500, 1000]
clip_at = float(sys.argv[11]) # random.choice([2,5,10,15])
assert clip_at == 15
clip_norm = int(sys.argv[12]) # random.choice([2,"infinity"])
assert clip_norm == 2
pos_embedding_size = int(sys.argv[13]) #random.choice([10,50, 100, 200])
assert pos_embedding_size in [10,50,100,200]
lstm_layers = int(sys.argv[14]) #random.choice([1,2,3])
assert lstm_layers in [1,2,3]
shallow = True if sys.argv[15] == "True" else False #random.choice([True, False])

rnn_dim = int(sys.argv[16]) #random.choice([128, 256, 300])
assert rnn_dim in [100, 200, 300]
bilinearSize = int(sys.argv[17]) if not shallow else 2*rnn_dim #random.choice([100, 300, 500]) 
useMean = False #random.choice([True, False])
input_dropoutRate = float(sys.argv[18]) #random.choice([0.0, 0.1])
assert input_dropoutRate in [0.0, 0.05, 0.1, 0.2]
labelMLPDimension = int(sys.argv[19]) #random.choice([100, 200, 300, 400])
assert labelMLPDimension in [100, 200, 300], labelMLPDimension


maxNumberOfUpdates = int(sys.argv[20]) if len(sys.argv) > 20 else 20000

model = sys.argv[21]
BASE_DIR = sys.argv[22]

FILE_NAME = "sososo_coarse_parser_plane"

assert len(sys.argv) == 23


names = ["rnn_dim", "lr_lm", "beta1", "beta2", "dropout_rate", "tokensPerBatch", "clip_at", "pos_embedding_size", "lstm_layers", "bilinearSize", "useMean", "shallow", "clip_norm", "input_dropoutRate", "labelMLPDimension", "lr_policy", "momentum_policy", "entropy_weight"]
params = [rnn_dim, lr_lm, beta1, beta2, dropout_rate, tokensPerBatch, clip_at, pos_embedding_size, lstm_layers, bilinearSize, useMean, shallow, clip_norm, input_dropoutRate, labelMLPDimension, lr_policy, momentum_policy, entropy_weight]

if prescribedID is not None and prescribedID != "NONE":
  myID = int(prescribedID)
else:
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


   # store new dependency links
   moved = [None] * len(sentence)
   for i, x in enumerate(linearized):
      moved[x["index"]-1] = i
   for i,x in enumerate(linearized):
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

assert model is not "RANDOM", "I think this shouldn't happen"

if model not in ["RANDOM", "RANDOM2", "RANDOM3", "RANDOM4", "RANDOM5"] and model != "REAL_REAL":
   temperature = 1.0
   inpModels_path = "../../../raw-results/"+"/"+BASE_DIR+"/"
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


vocab_size = 5

word_embedding_size = 0
pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = pos_embedding_size).cuda()



dropout = nn.Dropout(dropout_rate).cuda()

rnn = nn.LSTM(pos_embedding_size, rnn_dim, lstm_layers, bidirectional=True, batch_first=True).cuda() # word_embedding_size + (2*
for name, param in rnn.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)


headRep = nn.Linear(2*rnn_dim, bilinearSize).cuda()
depRep = nn.Linear(2*rnn_dim, bilinearSize).cuda()

headMLPOut = nn.Linear(bilinearSize, bilinearSize).cuda()
dependentMLPOut = nn.Linear(bilinearSize, bilinearSize).cuda()

labelMLP = nn.Linear(2*bilinearSize, labelMLPDimension).cuda()
labelDecoder = nn.Linear(labelMLPDimension, len(itos_pure_deps)+1).cuda()


U = nn.Parameter(torch.Tensor(bilinearSize,bilinearSize).cuda())

biasHead = nn.Parameter(torch.Tensor(1,bilinearSize,1).cuda())

components = [ pos_u_embeddings, rnn, headRep, depRep, headMLPOut, dependentMLPOut, labelMLP, labelDecoder] 

def parameters_lm():
 for c in components:
   for param in c.parameters():
      yield param
 yield U
 yield biasHead

from torch import optim

optimizer = optim.Adam(parameters_lm(), lr = lr_lm, betas=[beta1, beta2])

initrange = 0.01
pos_u_embeddings.weight.data.uniform_(-initrange, initrange)


U.data.fill_(0)
biasHead.data.fill_(0)


headMLPOut.bias.data.fill_(0)
headMLPOut.weight.data.uniform_(-initrange, initrange)

dependentMLPOut.bias.data.fill_(0)
dependentMLPOut.weight.data.uniform_(-initrange, initrange)

labelMLP.bias.data.fill_(0)
labelMLP.weight.data.uniform_(-initrange, initrange)

labelDecoder.bias.data.fill_(0)
labelDecoder.weight.data.uniform_(-initrange, initrange)




headRep.bias.data.fill_(0)
headRep.weight.data.uniform_(-initrange, initrange)

depRep.bias.data.fill_(0)
depRep.weight.data.uniform_(-initrange, initrange)

#
def prod(x):
   r = 1
   for s in x:
     r *= s
   return r

crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

import torch.cuda
import torch.nn.functional


inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)



content_pos = map(lambda x:stoi_pos_uni[x], filter(lambda y:y in itos_pos_uni, ["ADJ", "ADV", "NOUN", "NUM", "PROPN", "VERB"]))
function_pos = map(lambda x:stoi_pos_uni[x], filter(lambda y:y in itos_pos_uni,["ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT", "SCONJ", "SYM", "X"]))


baselinePerType = [4.0 for _ in itos_pure_deps]

def forward(current, computeAccuracy=False, doDropout=True):
       global biasHead
       global crossEntropy
       global batchSize
       batchSize = len(current)
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, batchOrdered)
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 , batchOrdered)) # if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0)

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()), Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()))
       loss = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
       optimizer.zero_grad()

       if True:
           pos_u_layer = pos_u_embeddings(Variable(torch.LongTensor(input_pos_u).transpose(0,1)).cuda())
           inputEmbeddings = pos_u_layer # torch.cat([pos_u_layer, pos_p_layer], dim=2) # words_layer, 
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              inputEmbeddings = dropout(inputEmbeddings)
           output, hidden = rnn(inputEmbeddings, hidden)

           outputFlat = output.contiguous().view(-1, 2*rnn_dim)
           if not shallow:
              heads = headRep(outputFlat)
              if doDropout:
                 heads = dropout(heads)
              heads = nn.ReLU()(heads)
              dependents = depRep(outputFlat)
              if doDropout:
                 dependents = dropout(dependents)
              dependents = nn.ReLU()(dependents)
           else:
              heads = outputFlat
              if doDropout:
                 heads = dropout(heads)
              dependents = outputFlat
              if doDropout:
                 dependents = dropout(dependents)


          
           heads = heads.view(batchSize, maxLength+2, 1, bilinearSize).contiguous() # .expand(batchSize, maxLength+2, maxLength+2, rnn_dim)
           dependents = dependents.view(batchSize, 1, maxLength+2, bilinearSize).contiguous() # .expand(batchSize, maxLength+2, maxLength+2, rnn_dim)
           
           part1 = torch.matmul(heads, U)
           bilinearAttention = torch.matmul(part1, torch.transpose(dependents, 2, 3)) # 

           heads = heads.view(-1, 1, bilinearSize)
         
           biasFromHeads = torch.matmul(heads,biasHead).view(batchSize, 1, 1, maxLength+2)
           bilinearAttention = bilinearAttention + biasFromHeads
           bilinearAttention = bilinearAttention.view(-1, maxLength+2)
           bilinearAttention = logsoftmax(bilinearAttention)
           bilinearAttention = bilinearAttention.view(batchSize, maxLength+2, maxLength+2)


           lossesHead = [[None]*batchSize for i in range(maxLength+1)]
           accuracy = 0
           accuracyLabeled = 0
           POSaccuracy = 0

           lossModuleTest = nn.NLLLoss(size_average=False, reduce=False , ignore_index=maxLength+1)
           lossModuleTestLabels = nn.NLLLoss(size_average=False, reduce=False , ignore_index=len(itos_pure_deps))


           targetTensor = [[maxLength+1 for _ in range(maxLength+2)] for _ in range(batchSize)]
 
           wordNum = 0
           for j in range(batchSize):
             for i in range(1,len(batchOrdered[j])+1):
               pos = input_pos_u[i][j]
               assert pos >= 3, (i,j)
               if False and pos-3 in function_pos:
                  continue
               else:
                  assert True or pos-3 in content_pos, (pos-3, itos_pos_uni[pos-3]) 
               if batchOrdered[j][i-1]["head"] == 0:
                  realHead = 0
               else:
                  realHead = batchOrdered[j][i-1]["reordered_head"]
               targetTensor[j][i] = realHead
               wordNum += 1

           if wordNum == 0:
                return 0, 0, 0, 0, wordNum

           targetTensorVariable = Variable(torch.LongTensor(targetTensor)).cuda()
           targetTensorLong = torch.LongTensor(targetTensor).cuda()
           lossesHead = lossModuleTest(bilinearAttention.view((batchSize)*(maxLength+2), maxLength+2), targetTensorLong.view((batchSize)*(maxLength+2)))
           lossesHead = lossesHead.view(batchSize, maxLength+2)
           if useMean:
              assert False
              loss += lossesHead.mean()
           else:
              loss += lossesHead.sum()
           lossWords += lossesHead.sum()

           heads = heads.view(batchSize, maxLength+2, bilinearSize)
           targetIndices = targetTensorVariable.unsqueeze(2).expand(batchSize, maxLength+2, bilinearSize)
           headStates = torch.gather(heads, 1, targetIndices)
           dependents = dependents.view(batchSize, maxLength+2, bilinearSize)
           headStates = headStates.view(batchSize, maxLength+2, bilinearSize)
           headsAndDependents = torch.cat([dependents, headStates], dim=2)
           labelHidden = labelMLP(headsAndDependents)
           if doDropout:
              labelHidden = dropout(labelHidden)
           labelHidden = nn.ReLU()(labelHidden)
           labelLogits = labelDecoder(labelHidden)
           labelSoftmax = logsoftmaxLabels(labelLogits)

           labelTargetTensor = [[len(itos_pure_deps) for _ in range(maxLength+2)] for _ in range(batchSize)]
           
           for j in range(batchSize):
             for i in range(1,len(batchOrdered[j])+1):
               pos = input_pos_u[i][j]
               assert pos >= 3, (i,j)
               if False and pos-3 in function_pos:
                  continue
               else:
                  assert True or pos-3 in content_pos  


               labelTargetTensor[j][i] = stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]
           labelTargetTensorVariable = Variable(torch.LongTensor(labelTargetTensor)).cuda()
   
           lossesLabels = lossModuleTestLabels(labelSoftmax.view((batchSize)*(maxLength+2), (1+len(itos_pure_deps))), labelTargetTensorVariable.view((batchSize)*(maxLength+2))).view(batchSize, maxLength+2)
           loss += lossesLabels.sum()

           lossesHeadsAndLabels = (lossesHead + lossesLabels).data.cpu().numpy()


           if computeAccuracy:
              wordNumAcc = 0
              for j in range(batchSize):
                  for i in range(1,len(batchOrdered[j])+1):
                       pos = input_pos_u[i][j]
                       assert pos >= 3, (i,j)
                       if False and pos-3 in function_pos:
                          continue
                       else:
                          assert True or pos-3 in content_pos  

                       predictions = bilinearAttention[j][i]
                       predictionsLabels = labelSoftmax[j][i]

                       if input_pos_u[i][j] == 0 or batchOrdered[j][i-1]["head"] == 0:
                          realHead = 0
                       else:
                          realHead = batchOrdered[j][i-1]["reordered_head"] # this starts at 1, so just right for the purposes here
                       predictedHead = np.argmax(predictions.data.cpu().numpy())
                       predictedLabel = np.argmax(predictionsLabels.data.cpu().numpy())
                       realLabel = labelTargetTensorVariable[j][i]
                       if input_pos_u[i][j] > 2:
                         accuracyLabeled += 1 if (predictedHead == realHead) and (predictedLabel == realLabel) else 0
                         accuracy += 1 if (predictedHead == realHead) else 0
                         wordNumAcc += 1
                       if printHere and j == batchSize/2: 
                          results = [j]
                          results.append(i)
                          results.append(itos[input_words[i][j]-3])
                          results.append(itos_pos_uni[input_pos_u[i][j]-3])
                          results.append(itos_pos_ptb[input_pos_p[i][j]-3])
                          results.append(lossesHead[j][i].data.cpu().numpy())
                          results.append(lossesLabels[j][i].data.cpu().numpy())
                          print results + [predictedHead, realHead, itos_pure_deps[predictedLabel] if predictedLabel < len(itos_pure_deps) else "--", itos_pure_deps[realLabel] if realLabel < len(itos_pure_deps) else "--"]
              assert wordNum == wordNumAcc, (wordNum, wordNumAcc)

       if printHere and wordNum > 0:
         if computeAccuracy:
            print "ACCURACY "+str(float(accuracy)/wordNum)+" "+str(float(POSaccuracy)/wordNum)
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy) if crossEntropy < 50 else "INFINITY"]
         print sys.argv
       if wordNum > 0:  
          crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum).data.cpu().numpy()
       policy_related_loss = 0
       return loss, policy_related_loss, accuracy if computeAccuracy else None, accuracyLabeled if computeAccuracy else None, wordNum

def backward(loss, policy_related_loss):
       if loss is 0:
          print "Absolutely Zero Loss"
          print current
          return

       loss.backward()

       torch.nn.utils.clip_grad_norm(parameters_lm(), clip_at, norm_type=clip_norm)
       optimizer.step()


def getPartitions(corpus):
  batch = list(corpus)
  batch = sorted(batch, key=len)
  partitions = [[]]
  lengthLast = 0
  for sentence in batch:
     if lengthLast > tokensPerBatch:
         partitions.append([])
         lengthLast = 0
     partitions[-1].append(sentence)
     lengthLast += len(sentence)
  shuffle(partitions)
  return partitions




devLosses = []
devAccuracies = []
devAccuraciesLabeled = []

import sys


def computeDevLoss():
         global printHere
         counterDev = 0
         corpusDev = CorpusIteratorFuncHead(language, "dev").iterator(rejectShortSentences = True)
         partitionsDev = getPartitions(corpusDev)
         devLoss = 0
         devAccuracy = 0
         devAccuracyLabeled = 0
         devWords = 0
         for partitionDev in partitionsDev:
              counterDev += 1
              printHere = (counterDev % 500 == 0)
              loss, _, accuracy, accuracyLabeled, wordNum = forward(partitionDev, computeAccuracy=True, doDropout=False)
              devLoss += loss.data.cpu().numpy()    
              devAccuracy += accuracy
              devAccuracyLabeled += accuracyLabeled
              devWords += wordNum
              if counterDev % 50 == 0:
                print "Run on dev "+str(counterDev)
                print (devLoss/devWords, float(devAccuracy)/devWords, float(devAccuracyLabeled)/devWords, devWords)

         newDevLoss = devLoss/devWords        
         newDevAccuracy = float(devAccuracy)/devWords
         newDevAccuracyLabeled = float(devAccuracyLabeled)/devWords
         devLosses.append(newDevLoss)
         devAccuracies.append(newDevAccuracy)
         devAccuraciesLabeled.append(newDevAccuracyLabeled)

counter = 0
epochs = 0
while True:
  corpus = CorpusIteratorFuncHead(language, "train").iterator(rejectShortSentences = True)
  partitions = getPartitions(corpus)
  epochs += 1
  for partition in partitions:
       if counter > maxNumberOfUpdates:
            print "Ran for a long time, quitting."
            quit()

       counter += 1
       printHere = (counter % 100 == 0)
       loss, policyLoss, _, _, wordNum = forward(partition)
       if wordNum == 0:
          assert loss is 0
       else:
          backward(loss, policyLoss)
       if printHere:
           print " ".join(map(str,[FILE_NAME, language, myID, counter, epochs, len(partitions), "MODEL", model ]))
           print zip(names, params)
           print devLosses
           print devAccuracies
           print devAccuraciesLabeled
       if counter % 500 == 0:
          print >> sys.stderr, (myID, "EPOCHS", epochs, "UPDATES", counter, "perEpoch", len(partitions), devLosses, devAccuracies, devAccuraciesLabeled)
          difference  = 0
          maxDifference = 0
          for i, key in enumerate(itos_deps):
             differenceHere = abs(distanceWeights[i] - originalDistanceWeights[key])
             difference += differenceHere
             maxDifference = max(differenceHere, maxDifference)
          print >> sys.stderr, (difference/len(itos_deps), maxDifference)



  if True: #counter % 5000 == 0:
         print >> sys.stderr, (myID, "EPOCHS", epochs, "UPDATES", counter)

         computeDevLoss()

         with open("../../../raw-results/parsing-upos/performance-"+language+"_"+FILE_NAME+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, " ".join(names)
              print >> outFile, " ".join(map(str,params))
              print >> outFile, " ".join(map(str,devLosses))
              print >> outFile, " ".join(map(str,devAccuracies))
              print >> outFile, " ".join(map(str,devAccuraciesLabeled))
              print >> outFile, " ".join(sys.argv)


         if devAccuracies[-1] == 0:
            print "Bad accuracy"
            quit()
         if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
            del devLosses[-1]
            print "Loss deteriorating, stop"
            quit()











 
