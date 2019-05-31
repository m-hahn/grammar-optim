# Optimizing a grammar for predictability





import random
import sys

objectiveName = "LM"


import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--language', type=str, dest="language")
parser.add_argument('--entropy_weight', type=float, default=0.001, dest="entropy_weight")
parser.add_argument('--lr_policy', type=float, default=0.001, dest="lr_policy")

parser.add_argument('--momentum_policy', type=float, default=0.9, dest="momentum_policy")
parser.add_argument('--lr_baseline_lm', type=float, default=1.0, dest="lr_baseline_lm")
parser.add_argument('--dropout_prob_lm', type=float, default=0.5, dest="dropout_prob_lm")
parser.add_argument('--lr_lm', type=float, default=0.1, dest="lr_lm")
parser.add_argument('--batchSize', type=int, default=1, dest="batchSize")
parser.add_argument('--lr_parser', type=float, dest="lr_parser")
parser.add_argument('--dropout_rate_parser', type=float, dest="dropout_rate_parser")
parser.add_argument('--beta1', type=float, dest="beta1")
parser.add_argument('--beta2', type=float, dest="beta2")
parser.add_argument('--clip_at_parser', type=float, dest="clip_at_parser")
parser.add_argument('--clip_norm_parser', type=int, dest="clip_norm_parser")
parser.add_argument('--pos_embedding_size_parser', type=int, dest="pos_embedding_size_parser")
parser.add_argument('--lstm_layers_parser', type=int, dest="lstm_layers_parser")
parser.add_argument('--shallow', type=bool, dest="shallow")
parser.add_argument('--rnn_dim_parser', type=int, dest="rnn_dim_parser")
parser.add_argument('--bilinearSize', type=int, dest="bilinearSize")
parser.add_argument('--input_dropoutRate_parser', type=float, dest="input_dropoutRate_parser")
parser.add_argument('--labelMLPDimension', type=int, dest="labelMLPDimension")


args = parser.parse_args()

if args.shallow:
   args.bilinearSize = 2*args.rnn_dim_parser





assert args.lr_policy < 1.0
assert args.momentum_policy < 1.0
assert args.entropy_weight >= 0
assert args.lr_parser < 0.1
assert args.beta1 == 0.9
assert args.beta2 in [0.9, 0.999]
assert args.dropout_rate_parser in [0.0, 0.1, 0.2, 0.3]
assert args.clip_at_parser == 15
assert args.clip_norm_parser == 2
assert args.pos_embedding_size_parser in [10,50,100,200]
assert args.lstm_layers_parser in [1,2,3]

assert args.rnn_dim_parser in [100, 200, 300]
assert args.input_dropoutRate_parser in [0.0, 0.05, 0.1, 0.2]
assert args.labelMLPDimension in [100, 200, 300], args.labelMLPDimension


maxNumberOfUpdates = int(sys.argv[20]) if len(sys.argv) > 20 else 20000

model = "REINFORCE"



names_parser = ["rnn_dim", "lr_lm", "beta1", "beta2", "dropout_rate",  "clip_at", "pos_embedding_size", "lstm_layers", "bilinearSize", "clip_norm", "input_dropoutRate", "labelMLPDimension", "lr_policy", "momentum_policy", "entropy_weight"]
params_parser = [args.rnn_dim_parser, args.lr_parser, args.beta1, args.beta2, args.dropout_rate_parser, args.clip_at_parser, args.pos_embedding_size_parser, args.lstm_layers_parser, args.bilinearSize, args.clip_norm_parser, args.input_dropoutRate_parser, args.labelMLPDimension, args.lr_policy, args.momentum_policy, args.entropy_weight]


myID = random.randint(0,10000000000)
random.seed(a=myID)

#



posUni = set() 
posFine = set() 



from math import log, exp, sqrt
from random import random, shuffle
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
     for sentence in CorpusIteratorFuncHead(args.language,partition).iterator():
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
   return dhLogits, vocab, keys, depsVocab

import torch.nn as nn
import torch
from torch.autograd import Variable

torch.manual_seed(myID)

# "linearization_logprobability"
def recursivelyLinearize(sentence, position, result, gradients_from_the_left_sum):
   line = sentence[position-1]
   # Loop Invariant: these are the gradients relevant at everything starting at the left end of the domain of the current element
   allGradients = gradients_from_the_left_sum + sum(line.get("children_decisions_logprobs",[]))

   if "linearization_logprobability" in line:
      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
   else:
      assert line["coarse_dep"] == "root"


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
import numpy as np

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()
logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = torch.cat([distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]].view(1) for x in remainingChildren])
           if reverseSoftmax:
              logits = -logits
           softmax = softmax_layer(logits.view(1,-1)).view(-1)
           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
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
      if line["coarse_dep"].startswith("punct"):
         continue
      key = line["coarse_dep"]
      line["dependency_key"] = key
      dhLogit = dhWeights[stoi_deps[key]]
      probability = 1/(1 + torch.exp(-dhLogit))
      dhSampled = (random() < probability.data.numpy())
      line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability.data.numpy())+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]].data.numpy())+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
      sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



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


relevantPath = "../../../raw-results/manual_output_funchead_langmod_coarse_best/"

import os
files = [x for x in os.listdir(relevantPath) if x.startswith(args.language+"_")]
posCount = 0
negCount = 0
for name in files:
  with open(relevantPath+name, "r") as inFile:
    for line in inFile:
        line = line.split("\t")
        if line[5] == "obj":
          dhWeight = float(line[4])
          if dhWeight < 0:
             negCount += 1
          elif dhWeight > 0:
             posCount += 1
          break

print(["Neg count", negCount, "Pos count", posCount])

if posCount >= 4 and negCount >= 4:
   print("Enough models!")
   quit()

with open("../../../raw-results/LOG"+args.language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
    print >> outFile, " ".join(sys.argv)
#

dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
for i, key in enumerate(itos_deps):
   dhLogits[key] = 0.0
   if key == "obj": 
       dhLogits[key] = (10.0 if posCount < negCount else -10.0)

   dhWeights.data[i] = dhLogits[key]

   originalDistanceWeights[key] = 0.0 #random()  
   distanceWeights.data[i] = originalDistanceWeights[key]



words = list(vocab.iteritems())
words = sorted(words, key = lambda x:x[1], reverse=True)
itos = map(lambda x:x[0], words)
stoi = dict(zip(itos, range(len(itos))))

if len(itos) > 6:
   assert stoi[itos[5]] == 5


vocab_size_parser = 5

vocab_size_lm = 50000







from torch import optim




#################################################################################3
# Language Model
#################################################################################3


# Initialize Components of Language Model Network
word_embeddings_lm = torch.nn.Embedding(num_embeddings = vocab_size_lm+3, embedding_dim = 50).cuda()
pos_u_embeddings_lm = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = 10).cuda()
pos_p_embeddings_lm = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=10).cuda()

# Initialize Control Variate for Predictability
baseline_lm = torch.nn.Embedding(num_embeddings = vocab_size_lm+3, embedding_dim=1).cuda()
baseline_upos_lm = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim=1).cuda()
baseline_ppos_lm = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=1).cuda()


dropout_lm = nn.Dropout(args.dropout_prob_lm).cuda()

rnn_lm = nn.LSTM(70, 128, 2).cuda()
for name, param in rnn_lm.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder_lm = nn.Linear(128,vocab_size_lm+3).cuda()
pos_ptb_decoder_lm = nn.Linear(128,len(posFine)+3).cuda()

components_lm = [word_embeddings_lm, pos_u_embeddings_lm, pos_p_embeddings_lm, rnn_lm, decoder_lm, pos_ptb_decoder_lm, baseline_lm, baseline_upos_lm, baseline_ppos_lm]

def parameters_lm():
 for c in components_lm:
   for param in c.parameters():
      yield param

parameters_lm_cached = [x for x in parameters_lm()]


# Initialize Language Model Parameter

initrange = 0.1
word_embeddings_lm.weight.data.uniform_(-initrange, initrange)
pos_u_embeddings_lm.weight.data.uniform_(-initrange, initrange)
pos_p_embeddings_lm.weight.data.uniform_(-initrange, initrange)
decoder_lm.bias.data.fill_(0)
decoder_lm.weight.data.uniform_(-initrange, initrange)
pos_ptb_decoder_lm.bias.data.fill_(0)
pos_ptb_decoder_lm.weight.data.uniform_(-initrange, initrange)
baseline_lm.weight.data.fill_(0) #uniform_(-initrange, initrange)
baseline_upos_lm.weight.data.fill_(0) #uniform_(-initrange, initrange)
baseline_ppos_lm.weight.data.fill_(0) #uniform_(-initrange, initrange)




#################################################################################3
# Grammar Parameter
#################################################################################3


def parameters_policy():
 yield dhWeights
 yield distanceWeights



parameters_policy_cached = [x for x in parameters_policy()]

#################################################################################3
#################################################################################3
#################################################################################3

# Keep running averages of surprisal losses for inspection
crossEntropy_lm = 10.0

# Encode word as an integer
def encodeWord_lm(w):
   return stoi[w]+3 if stoi[w] < vocab_size_lm else 1

import torch.cuda
import torch.nn.functional



# Control variates
baselineAverageLoss_lm = 0

# Counts gradient steps so far
counter = 0

# Losses on development set for Early Stopping
lastDevLoss_lm = None

failedDevRuns_lm = 0

devLosses_lm = [] 

loss_op = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index = 0)



def doForwardPass(current, train=True):
       global biasHead
       batchSize = len(current)
       global counter
       global crossEntropy_lm

       global printHere
       global devLosses_lm
       global baselineAverageLoss_lm
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, batchOrdered)
       # current is already sorted by length
       maxLength = max(lengths)
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord_lm(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0), batchOrdered))


       ###################################################################
       # Zero Gradients
       ###################################################################


       for c in components_lm:
          c.zero_grad()


       for p in  [dhWeights, distanceWeights]:
          if p.grad is not None:
             p.grad.data = p.grad.data.mul(args.momentum_policy)








       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss_lm = 0

       wordNum_lm = 0
       lossWords_lm = 0

       policyGradientLoss_lm = 0

       baselineLoss_lm = 0



       #############################################################################
       # Language Model 
       #############################################################################


       totalQuality = 0.0

       if True:
           wordIndices = Variable(torch.LongTensor(input_words)).cuda()
           pos_p_indices = Variable(torch.LongTensor(input_pos_p)).cuda()
           words_layer = word_embeddings_lm(wordIndices)
           pos_u_indices = Variable(torch.LongTensor(input_pos_u)).cuda()
           pos_u_layer = pos_u_embeddings_lm(pos_u_indices)
           pos_p_layer = pos_p_embeddings_lm(pos_p_indices)
           input_concat = torch.cat([words_layer, pos_u_layer, pos_p_layer], dim=2)
           inputEmbeddings = dropout_lm(input_concat) if train else input_concat

           output, hidden = rnn_lm(inputEmbeddings, None)
           baseline_predictions = log(vocab_size_lm) + log(len(posFine)) + baseline_lm(wordIndices) + baseline_upos_lm(pos_u_indices) + baseline_ppos_lm(pos_p_indices)

           droppedOutput = dropout_lm(output) if train else output

          # word logits
           word_logits = decoder_lm(droppedOutput)
           word_logits = word_logits.view(-1, vocab_size_lm+3)
           word_softmax = logsoftmax(word_logits)
           word_softmax = word_softmax.view(-1, args.batchSize, vocab_size_lm+3)

           # pos logits
           pos_logits = pos_ptb_decoder_lm(droppedOutput)
           pos_logits = pos_logits.view(-1, len(posFine)+3)
           pos_softmax = logsoftmax(pos_logits)
           pos_softmax = pos_softmax.view(-1, args.batchSize, len(posFine)+3)

           lossesWord = [[None]*args.batchSize for i in range(maxLength+1)]
           lossesPOS = [[None]*args.batchSize for i in range(maxLength+1)]

           lossesWord_tensor = loss_op(word_softmax.view(-1, vocab_size_lm+3)[:-1], wordIndices[1:].view(-1)).view(-1, args.batchSize)
           lossesPOS_tensor = loss_op(pos_softmax.view(-1, len(posFine)+3)[:-1], pos_p_indices[1:].view(-1)).view(-1, args.batchSize)

           reward = (lossesWord_tensor + lossesPOS_tensor).detach()

           baseline_shifted = baseline_predictions[1:]

           baselineLoss_lm = torch.nn.functional.mse_loss(baseline_shifted.view(-1, args.batchSize), reward.view(-1, args.batchSize), size_average=False, reduce=False)

           baselineAverageLoss_lm = 0.99 * baselineAverageLoss_lm + (1-0.99) * baselineLoss_lm.cpu().data.mean().numpy()
           if printHere:
              print(baselineLoss_lm)
              print(["Baseline loss", sqrt(baselineAverageLoss_lm)])

           rewardMinusBaseline_lm = (reward.view(-1, args.batchSize) - baseline_shifted.view(-1, args.batchSize)).detach().cpu().data.numpy()

           for i in range(0,len(input_words)-1): 
              for j in range(args.batchSize):
                 if input_words[i+1][j] != 0:
                    policyGradientLoss_lm += (float(rewardMinusBaseline_lm[i][j]) * batchOrdered[j][i]["relevant_logprob_sum"])
                    if input_words[i+1] > 2 and j == 0 and printHere:
                       print [itos[input_words[i+1][j]-3], itos_pos_ptb[input_pos_p[i+1][j]-3], lossesWord_tensor[i][j].data.cpu().numpy(), lossesPOS_tensor[i][j].data.cpu().numpy(), baseline_predictions[i+1][j].data.cpu().numpy()]
                    wordNum_lm += 1

       lossWords_lm = lossesWord_tensor.sum()
       lossPOS_lm = lossesPOS_tensor.sum()
       loss_lm += lossWords_lm
       loss_lm += lossPOS_lm
       if wordNum_lm == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss_lm/wordNum_lm
         print lossWords_lm/wordNum_lm
         print ["CROSS ENTROPY LM", crossEntropy_lm, exp(crossEntropy_lm)]
         print baselineAverageLoss_lm
       crossEntropy_lm = 0.99 * crossEntropy_lm + 0.01 * (lossWords_lm/wordNum_lm).data.cpu().numpy()
       totalQuality_lm = loss_lm.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords_lm = wordNum_lm

       policy_related_loss_lm =  policyGradientLoss_lm # lives on CPU
       return (loss_lm, baselineLoss_lm, policy_related_loss_lm, totalQuality_lm, numberOfWords_lm)


def  doBackwardPass(loss_lm, baselineLoss_lm, policy_related_loss_lm):
       if printHere:
         print "BACKWARD 1"

       # Objective function for grammar
       policy_related_loss = policy_related_loss_lm

       global dhWeights

       # Entropy Regularization
       probabilities = torch.sigmoid(dhWeights)
       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))
       policy_related_loss += args.entropy_weight * neg_entropy # lives on CPU

       # Backprop for grammar parameters
       policy_related_loss.backward()
       if printHere:
         print "BACKWARD 2"

       totalLoss = loss_lm
       totalLoss += args.lr_baseline_lm * baselineLoss_lm.sum()
       totalLoss.backward() # lives on GPU



       if printHere:
         print args 
         print "BACKWARD 3 "+__file__+" "+args.language+" "+str(myID)+" "+str(counter)+" "+str(lastDevLoss_lm)+" "+str(failedDevRuns_lm)+"  "+(" ".join(map(str,["ENTROPY", args.entropy_weight, "LR_POLICY", args.lr_policy, "MOMENTUM", args.momentum_policy])))
         print "dev losses LM"
         print devLosses_lm
         print crossEntropy_lm

       # Gradient clipping for language model
       torch.nn.utils.clip_grad_norm(parameters_lm_cached, 5.0, norm_type='inf')
       counterHere = 0


       for param in parameters_lm_cached:
         counterHere += 1
         if param.grad is None:
           assert False
         param.data.sub_(args.lr_lm * param.grad.data)





       for param in parameters_policy_cached:
         counterHere += 1
         if counter < 200 and (param is distanceWeights or param is dhWeights): # allow baseline to warum up
             continue
         if param.grad is None:
           print counterHere
           print "WARNING: None gradient"
           continue
         param.data.sub_(args.lr_policy * param.grad.data)



def computeDevLoss():
   global printHere
   global counter


   devLoss_lm = 0.0
   devWords_lm = 0



   corpusDev = CorpusIteratorFuncHead(args.language,"dev").iterator(rejectShortSentences = True)

   while True:
     try:
        batch = map(lambda x:next(corpusDev), 10*range(args.batchSize))
     except StopIteration:
        break
     batch = sorted(batch, key=len)
     partitions = range(10)
     shuffle(partitions)
     for partition in partitions:
        counter += 1
        printHere = (counter % 50 == 0)
        current = batch[partition*args.batchSize:(partition+1)*args.batchSize]
 
        fromLM = doForwardPass(current, train=False)
        _, _, _, newLoss_lm, newWords_lm = fromLM


        devLoss_lm += newLoss_lm
        devWords_lm += newWords_lm



   return devLoss_lm/devWords_lm

# Training Loop
while True:
  corpus = CorpusIteratorFuncHead(args.language).iterator(rejectShortSentences = True)


  while True:
    try:
       batch = map(lambda x:next(corpus), 10*range(args.batchSize))
    except StopIteration:
       break
    batch = sorted(batch, key=len)
    partitions = range(10)
    shuffle(partitions)
    for partition in partitions:
       counter += 1
       printHere = (counter % 100 == 0)
       current = batch[partition*args.batchSize:(partition+1)*args.batchSize]

       # Run model on syntactic tree
       fromLM = doForwardPass(current)
       loss_lm, baselineLoss_lm, policy_related_loss_lm, _, wordNumInPass_lm = fromLM

       # Update model parameters
       if wordNumInPass_lm > 0:
         doBackwardPass(loss_lm, baselineLoss_lm, policy_related_loss_lm)
       else: # In case a sentence is empty (which can happen if it consists entirely of punctuation), skip.
         print "No words, skipped backward"
       # In intervals of 50,000 gradient steps, run on held-out set, and decide whether to stop optimizing.
       if counter % 50000 == 0:
          newDevLoss_lm = computeDevLoss()
          devLosses_lm.append(newDevLoss_lm)
          print "New dev loss LM     "+str(newDevLoss_lm)+". previous was: "+str(lastDevLoss_lm)

          if lastDevLoss_lm is None or newDevLoss_lm < lastDevLoss_lm:
             lastDevLoss_lm = newDevLoss_lm
             failedDevRuns_lm = 0
          else:
             failedDevRuns_lm += 1
             print devLosses_lm



          # Saving grammar parameters to file
          print "Saving"
          save_path = "../../../raw-results/"
          with open(save_path+"/manual_output_funchead_langmod_coarse_best/"+args.language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss_LM", "DH_Weight","CoarseDependency","DistanceWeight", "EntropyWeight", "ObjectiveName"]))
             for i in range(len(itos_deps)):
                key = itos_deps[i]
                dhWeight = dhWeights[i].data.numpy()
                distanceWeight = distanceWeights[i].data.numpy()
                dependency = key
                print >> outFile, "\t".join(map(str,[myID, __file__, counter, devLosses_lm[-1], dhWeight, dependency, distanceWeight, args.entropy_weight, objectiveName]))

          # Stop optimization if held-out losses are not going down any more
          if failedDevRuns_lm >0:
              quit()



