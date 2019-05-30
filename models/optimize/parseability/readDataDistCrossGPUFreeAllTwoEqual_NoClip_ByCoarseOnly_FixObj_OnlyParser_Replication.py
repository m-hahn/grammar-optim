#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import time

backwardAll = [0]


# TODO also try other optimizers


# parser from readDataDistEnglishGPUFreeParsingUnbiasedEntropy_Fixed_REAL_Fast_Broad_Config_Dev_Labeled_MLP_Batch_ContentOnly_Controlled_Reinforce_AllWords_Zero_FuncHead.py
# DepL from readDataDistCrossLGPUDepLengthMomentumEntropyUnbiasedBaseline_OrderBugFixed_NoPunct_NEWPYTORCH_AllCorpPerLang_BoundIterations_FuncHead.py

import random
import sys

objectiveName = "ParserCoarse"


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
parser.add_argument('--maxNumberEpochs', type=int, dest="maxNumberEpochs", default=100000)



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

# English RANDOM_BY_TYPE 0.002 0.9 0.9 0.3 1000 15 50 3 True 300 2 0.05 200 300 742002619


model = "REINFORCE"



names_parser = ["rnn_dim", "lr_lm", "beta1", "beta2", "dropout_rate",  "clip_at", "pos_embedding_size", "lstm_layers", "bilinearSize", "clip_norm", "input_dropoutRate", "labelMLPDimension", "lr_policy", "momentum_policy", "entropy_weight"]
params_parser = [args.rnn_dim_parser, args.lr_parser, args.beta1, args.beta2, args.dropout_rate_parser, args.clip_at_parser, args.pos_embedding_size_parser, args.lstm_layers_parser, args.bilinearSize, args.clip_norm_parser, args.input_dropoutRate_parser, args.labelMLPDimension, args.lr_policy, args.momentum_policy, args.entropy_weight]


myID = random.randint(0,10000000)
#
with open("/juicier/scr120/scr/mhahn/deps/LOG"+args.language+"_"+__file__+"_model_"+str(myID)+".txt", "w") as outFile:
    print >> outFile, " ".join(sys.argv)
#


posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]

deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp, sqrt
from random import random, shuffle
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
#   print ["DECISIONS_PREPARED", line["index"], line["word"], line["dep"], line["head"], allGradients.data.numpy()]
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
           #print logits
           if reverseSoftmax:
              logits = -logits
           #print (reverseSoftmax, logits)
           softmax = softmax_layer(logits.view(1,-1)).view(-1)
           selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
       return childrenLinearized           
#           softmax = torch.distributions.Categorical(logits=logits)
#           selected = softmax.sample()
#           print selected
#           quit()
#           softmax = torch.cat(logits)



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
      probability = 1/(1 + torch.exp(-dhLogit))
      dhSampled = (random() < probability.data.numpy())
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      line["ordering_decision_log_probability"] = torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy(),
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

#         shuffle(line["children_HD"])
   
   linearized = []
   recursivelyLinearize(sentence, root, linearized, Variable(torch.FloatTensor([0.0])))
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

dhWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
distanceWeights = Variable(torch.FloatTensor([0.0] * len(itos_deps)), requires_grad=True)
for i, key in enumerate(itos_deps):

   # take from treebank, or randomize
   dhLogits[key] = 0.0 #2*(random()-0.5)
   if key == "obj": # this being the reference point, we fix this
       dhLogits[key] = (10.0 if random() > 0.5 else -10.0)

   dhWeights.data[i] = dhLogits[key]

   originalDistanceWeights[key] = 0.0 #random()  
   distanceWeights.data[i] = originalDistanceWeights[key]



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

vocab_size_parser = 5

vocab_size_lm = 50000

#################################################################################3
# Parser
#################################################################################3

word_embedding_size_parser = 0
# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = word_embedding_size).cuda()
pos_u_embeddings_parser = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = args.pos_embedding_size_parser).cuda()
pos_p_embeddings_parser = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=args.pos_embedding_size_parser).cuda()

dropout_parser = nn.Dropout(args.dropout_rate_parser).cuda()

rnn_parser = nn.LSTM(args.pos_embedding_size_parser, args.rnn_dim_parser, args.lstm_layers_parser, bidirectional=True, batch_first=True).cuda() # word_embedding_size + (2*
for name, param in rnn_parser.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)


headRep = nn.Linear(2*args.rnn_dim_parser, args.bilinearSize).cuda()
depRep = nn.Linear(2*args.rnn_dim_parser, args.bilinearSize).cuda()

headMLPOut = nn.Linear(args.bilinearSize, args.bilinearSize).cuda()
dependentMLPOut = nn.Linear(args.bilinearSize, args.bilinearSize).cuda()

labelMLP = nn.Linear(2*args.bilinearSize, args.labelMLPDimension).cuda()
labelDecoder = nn.Linear(args.labelMLPDimension, len(itos_pure_deps)+1).cuda()

#bilinearMatrix = nn.Bilinear(rnn_dim, rnn_dim, 1).cuda()

U = nn.Parameter(torch.Tensor(args.bilinearSize,args.bilinearSize).cuda())


biasHead = nn.Parameter(torch.Tensor(1,args.bilinearSize,1).cuda())
##biasDependent = nn.Parameter(torch.Tensor(args.bilinearSize).cuda())
#
#
#biasLabel = Variable(torch.Tensor((1+len(itos_pure_deps))).cuda(), requires_grad=True)
#U_label = Variable(torch.Tensor((1+len(itos_pure_deps)), args.bilinearSize,args.bilinearSize).cuda(), requires_grad=True)
#linear_Label_dep = Variable(torch.Tensor(args.bilinearSize, (1+len(itos_pure_deps))).cuda(), requires_grad=True)
#linear_Label_head = Variable(torch.Tensor(args.bilinearSize, (1+len(itos_pure_deps))).cuda(), requires_grad=True)
#


components_parser = [ pos_u_embeddings_parser, rnn_parser, headRep, depRep, headMLPOut, dependentMLPOut, labelMLP, labelDecoder] # baseline, word_embeddings, pos_p_embeddings, 
def parameters_parser():
 for c in components_parser:
   for param in c.parameters():
      yield param
 yield U
 yield biasHead
# yield biasLabel
# yield U_label
# yield linear_Label_dep
# yield linear_Label_head







from torch import optim

optimizer_parser = optim.Adam(parameters_parser(), lr = args.lr_parser, betas=[args.beta1, args.beta2])
#optimizer = optim.SGD(parameters_lm(), lr = 0.0002, momentum=0.9) #, betas=[0.9, 0.9])
#optimizer_policy = optim.SGD(parameters_ordering(), lr = lr_policy, momentum=momentum_policy)

initrange = 0.01
#word_embeddings.weight.data.uniform_(-initrange, initrange)
pos_u_embeddings_parser.weight.data.uniform_(-initrange, initrange)
pos_p_embeddings_parser.weight.data.uniform_(-initrange, initrange)

U.data.fill_(0)
biasHead.data.fill_(0)
#biasLabel.data.fill_(0)
#U_label.data.fill_(0)
#linear_Label_dep.data.fill_(0)
#linear_Label_head.data.fill_(0)


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

#bilinearMatrix.bias.data.fill_(0)
#bilinearMatrix.weight.data.uniform_(-0.01, 0.01)


#baseline.bias.data.fill_(0)
#baseline.weight.data.uniform_(-initrange, initrange)





#################################################################################3
# Policy
#################################################################################3


def parameters_policy():
 yield dhWeights
 yield distanceWeights



parameters_policy_cached = [x for x in parameters_policy()]

#################################################################################3
#################################################################################3
#################################################################################3


crossEntropy_parser = 10.0


def encodeWord_lm(w):
   return stoi[w]+3 if stoi[w] < vocab_size_lm else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)



import torch.cuda
import torch.nn.functional



inputDropout_parser = torch.nn.Dropout2d(p=args.input_dropoutRate_parser)



content_pos = map(lambda x:stoi_pos_uni[x], filter(lambda y:y in itos_pos_uni, ["ADJ", "ADV", "NOUN", "NUM", "PROPN", "VERB"]))
function_pos = map(lambda x:stoi_pos_uni[x], filter(lambda y:y in itos_pos_uni,["ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT", "SCONJ", "SYM", "X"]))




baselinePerType_parser = [4.0 for _ in itos_pure_deps]

counter = 0


lastDevLoss_parser = None

failedDevRuns_parser = 0

devLosses_parser = [] 

loss_op = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index = 0)



def doForwardPass(current, train=True, computeAccuracy_parser=False, doDropout_parser=True):
       assert train == doDropout_parser
       assert train == (not computeAccuracy_parser)
       global biasHead
       batchSize = len(current)
       global counter
       global crossEntropy_parser

       global printHere
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


       optimizer_parser.zero_grad()

       for p in  [dhWeights, distanceWeights]:
          if p.grad is not None:
             p.grad.data = p.grad.data.mul(args.momentum_policy)








       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()), Variable(torch.FloatTensor().new(2, batchSize, 128).zero_()))
       loss_parser = 0

       lossWords_parser = 0

       policyGradientLoss_parser = 0

       baselineLoss_parser = 0


       #############################################################################
       # Parser
       #############################################################################

       if True:
           # TODO word dropout could also be added: randomly sprinkle `1' (UNK) in the LongTensor (not into input_words -- this will also be used for the softmax!)
#           words_layer = word_embeddings(Variable(torch.LongTensor(input_words)).cuda())
           pos_u_layer_parser = pos_u_embeddings_parser(Variable(torch.LongTensor(input_pos_u).transpose(0,1)).cuda())
#           pos_p_layer = pos_p_embeddings(Variable(torch.LongTensor(input_pos_p)).cuda())
           inputEmbeddings_parser = pos_u_layer_parser # torch.cat([pos_u_layer, pos_p_layer], dim=2) # words_layer, 
           if doDropout_parser:
              inputEmbeddings_parser = inputDropout_parser(inputEmbeddings_parser)
 #             if printHere:
 #               print inputEmbeddings
              inputEmbeddings_parser = dropout_parser(inputEmbeddings_parser)
           output_parser, hidden_parser = rnn_parser(inputEmbeddings_parser, None)

           outputFlat_parser = output_parser.contiguous().view(-1, 2*args.rnn_dim_parser)
           if not args.shallow:
              heads = headRep(outputFlat_parser)
              if doDropout_parser:
                 heads = dropout_parser(heads)
              heads = nn.ReLU()(heads)
              dependents = depRep(outputFlat)
              if doDropout_parser:
                 dependents = dropout_parser(dependents)
              dependents = nn.ReLU()(dependents)
           else:
              heads = outputFlat_parser
              if doDropout_parser:
                 heads = dropout_parser(heads)
              dependents = outputFlat_parser
              if doDropout_parser:
                 dependents = dropout_parser(dependents)

           heads = heads.view(args.batchSize, maxLength+2, 1, args.bilinearSize).contiguous() # .expand(batchSize, maxLength+2, maxLength+2, rnn_dim)
           dependents = dependents.view(args.batchSize, 1, maxLength+2, args.bilinearSize).contiguous() # .expand(batchSize, maxLength+2, maxLength+2, rnn_dim)
           
           part1 = torch.matmul(heads, U)
           bilinearAttention = torch.matmul(part1, torch.transpose(dependents, 2, 3)) # 

#           biasHead = biasHead.view(1,args.bilinearSize,1) #.unsqueeze(0).unsqueeze(0)
           heads = heads.view(-1, 1, args.bilinearSize)
         
           biasFromHeads = torch.matmul(heads,biasHead).view(args.batchSize, 1, 1, maxLength+2)
           bilinearAttention = bilinearAttention + biasFromHeads


           bilinearAttention = bilinearAttention.view(-1, maxLength+2)
           bilinearAttention = logsoftmax(bilinearAttention)
           bilinearAttention = bilinearAttention.view(args.batchSize, maxLength+2, maxLength+2)

           lossesHead = [[None]*args.batchSize for i in range(maxLength+1)]
           accuracy = 0
           accuracyLabeled = 0
           POSaccuracy = 0

           lossModuleTest = nn.NLLLoss(size_average=False, reduce=False , ignore_index=maxLength+1)
           lossModuleTestLabels = nn.NLLLoss(size_average=False, reduce=False , ignore_index=len(itos_pure_deps))

           targetTensor = [[maxLength+1 for _ in range(maxLength+2)] for _ in range(args.batchSize)]
 
           wordNum_parser = 0
           for j in range(args.batchSize):
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
                  realHead = batchOrdered[j][i-1]["reordered_head"] # this starts at 1, so just right for the purposes here
               targetTensor[j][i] = realHead
               wordNum_parser += 1

           if wordNum_parser == 0:
                return 0, 0, 0, 0, wordNum_parser

           targetTensorVariable = Variable(torch.LongTensor(targetTensor)).cuda()
           targetTensorLong = torch.LongTensor(targetTensor).cuda()
           lossesHead = lossModuleTest(bilinearAttention.view((args.batchSize)*(maxLength+2), maxLength+2), targetTensorLong.view((args.batchSize)*(maxLength+2)))
           lossesHead = lossesHead.view(args.batchSize, maxLength+2)
           loss_parser += lossesHead.sum()
           lossWords_parser += lossesHead.sum()

           heads = heads.view(args.batchSize, maxLength+2, args.bilinearSize)
           targetIndices = targetTensorVariable.unsqueeze(2).expand(args.batchSize, maxLength+2, args.bilinearSize)
           headStates = torch.gather(heads, 1, targetIndices)
           dependents = dependents.view(args.batchSize, maxLength+2, args.bilinearSize)
           headStates = headStates.view(args.batchSize, maxLength+2, args.bilinearSize)
           headsAndDependents = torch.cat([dependents, headStates], dim=2)
           labelHidden = labelMLP(headsAndDependents)
           if doDropout_parser:
              labelHidden = dropout_parser(labelHidden)
           labelHidden = nn.ReLU()(labelHidden)
           labelLogits = labelDecoder(labelHidden)
           labelSoftmax = logsoftmaxLabels(labelLogits)

           labelTargetTensor = [[len(itos_pure_deps) for _ in range(maxLength+2)] for _ in range(args.batchSize)]
           
           for j in range(args.batchSize):
             for i in range(1,len(batchOrdered[j])+1):
               pos = input_pos_u[i][j]
               assert pos >= 3, (i,j)
               if False and pos-3 in function_pos:
                  continue
               else:
                  assert True or pos-3 in content_pos  

               labelTargetTensor[j][i] = stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]
           labelTargetTensorVariable = Variable(torch.LongTensor(labelTargetTensor)).cuda()
   
           lossesLabels = lossModuleTestLabels(labelSoftmax.view((args.batchSize)*(maxLength+2), (1+len(itos_pure_deps))), labelTargetTensorVariable.view((args.batchSize)*(maxLength+2))).view(args.batchSize, maxLength+2)
           loss_parser += lossesLabels.sum()

           lossesHeadsAndLabels = (lossesHead + lossesLabels).data.cpu().numpy()
           policyGradientLoss_parser = 0
           for j in range(args.batchSize):
             lossForThisSentenceMinusBaselines_parser = 0
             for i in range(1,len(batchOrdered[j])+1):
               pos = input_pos_u[i][j]
               assert pos >= 3, (i,j)
               if False and pos-3 in function_pos:
                  continue
               else:
                  assert True or pos-3 in content_pos  
               lossForThisSentenceMinusBaselines_parser += (lossesHeadsAndLabels[j][i]  - log(len(batchOrdered[j])) - baselinePerType_parser[stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]])
               if printHere:
                  print ("parsing", lossesHeadsAndLabels[j][i], baselinePerType_parser[stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]] + log(len(batchOrdered[j])), (lossesHeadsAndLabels[j][i]  - log(len(batchOrdered[j])) - baselinePerType_parser[stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]]))
             policyGradientLoss_parser += batchOrdered[j][-1]["relevant_logprob_sum"] * lossForThisSentenceMinusBaselines_parser

           for j in range(args.batchSize):
             for i in range(1,len(batchOrdered[j])+1):
               baselinePerType_parser[stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]] = 0.99 * baselinePerType_parser[stoi_pure_deps[batchOrdered[j][i-1]["coarse_dep"]]] + (1-0.99) * (lossesHeadsAndLabels[j][i] - log(len(batchOrdered[j])))



           if computeAccuracy_parser:
              wordNumAcc = 0
              for j in range(args.batchSize):
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
                       if (True or input_words[i] > 2) and printHere and j == args.batchSize/2: 
                          results = [j]
                          results.append(i)
                          results.append(itos[input_words[i][j]-3])
                          results.append(itos_pos_uni[input_pos_u[i][j]-3])
                          results.append(itos_pos_ptb[input_pos_p[i][j]-3])
                          results.append(lossesHead[j][i].data.cpu().numpy())
                          results.append(lossesLabels[j][i].data.cpu().numpy())
                          print results + [predictedHead, realHead, itos_pure_deps[predictedLabel] if predictedLabel < len(itos_pure_deps) else "--", itos_pure_deps[realLabel] if realLabel < len(itos_pure_deps) else "--"]
              assert wordNum_parser == wordNumAcc, (wordNum_parser, wordNumAcc)

       if printHere and wordNum_parser > 0:
         if computeAccuracy_parser:
            print "ACCURACY "+str(float(accuracy)/wordNum_parser)+" "+str(float(POSaccuracy)/wordNum_parser)
         print loss_parser/wordNum_parser
         print lossWords_parser/wordNum_parser
         print ["CROSS ENTROPY Parser", crossEntropy_parser, exp(crossEntropy_parser) if crossEntropy_parser < 50 else "INFINITY"]
       if wordNum_parser > 0:  
          crossEntropy_parser = 0.99 * crossEntropy_parser + 0.01 * (lossWords_parser/wordNum_parser).data.cpu().numpy()



#       probabilities = torch.sigmoid(dhWeights)
#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))
       policy_related_loss_parser = policyGradientLoss_parser # lives on CPU
       # removed args.entropy_weight * neg_entropy


       return (loss_parser, policy_related_loss_parser, accuracy if computeAccuracy_parser else None, accuracyLabeled if computeAccuracy_parser else None, wordNum_parser)

backPropTime = 0.0

def  doBackwardPass(loss_parser, policy_related_loss_parser):
       global printHere
       if printHere:
         print "BACKWARD 1"
       policy_related_loss = policy_related_loss_parser

       global dhWeights
       probabilities = torch.sigmoid(dhWeights)
       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))
       policy_related_loss += args.entropy_weight * neg_entropy # lives on CPU


       policy_related_loss.backward()
       if printHere:
         print "BACKWARD 2"

       #totalLoss = loss_lm
       #totalLoss += args.lr_baseline_lm * baselineLoss_lm.sum()
       #totalLoss += loss_parser
       #totalLoss.backward() # lives on GPU

       if counter % 10000 == 0:
            backwardAll.append(0)
       while int(counter/10000)>= len(backwardAll):
          backwardAll.append(0)


       beforeBack = time.time()




       totalLoss = loss_parser
       totalLoss.backward() # lives on GPU
       backwardAll[int(counter/10000)] += (time.time() - beforeBack)/100


       if printHere:
         print args 
         print "BACKWARD 3 "+__file__+" "+args.language+" "+str(myID)+" "+str(counter)+" "+"  "+(" ".join(map(str,["ENTROPY", args.entropy_weight, "LR_POLICY", args.lr_policy, "MOMENTUM", args.momentum_policy])))
         print "dev losses parser"
         print devLosses_parser
         print crossEntropy_parser
         print "dev accuracies parser"
         print devAccuracies_parser
         print "dev accuracies labeled parser"
         print devAccuraciesLabeled_parser
         print backwardAll

#       print("Before")
   #    if printHere:
    #      print([counter, "Distance grad", torch.abs(distanceWeights.grad).sum()])
       if random() > 0.99:
          print("INFINITY NORM OF GRADIENTS", max(p.grad.data.abs().max() for p in parameters_policy_cached))
#       if len([x for x in parameters_policy_cached if x.grad is not None]) > 0:
#          torch.nn.utils.clip_grad_norm(parameters_policy_cached, 5.0, norm_type='inf')
#       else:
#         print "Skipping gradient clipping for policy"
       counterHere = 0

       # TODO the step for the parser


       optimizer_parser.step()


       for param in parameters_policy_cached:
         counterHere += 1
         if counter < 200 and (param is distanceWeights or param is dhWeights): # allow baseline to warum up
             continue
         if param.grad is None:
           print counterHere
           print "WARNING: None gradient"
           continue
         param.data.sub_(args.lr_policy * param.grad.data)

devAccuracies_parser = []
devAccuraciesLabeled_parser = []


def computeDevLoss():
   global printHere
   global counter
   global devAccuracies_parser
   global devAccuraciesLabeled_parser



   devLoss_parser = 0.0
   devWords_parser = 0
   devAccuracy_parser = 0
   devAccuracyLabeled_parser = 0


#   corpusDev = getNextSentence("dev")
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
 
        fromParser = doForwardPass(current, train=False, computeAccuracy_parser=True, doDropout_parser=False)
        loss_parser, _, accuracy_parser ,accuracyLabeled_parser, wordNum_parser = fromParser



        devLoss_parser += loss_parser.data.cpu().numpy()    
        devAccuracy_parser += accuracy_parser
        devAccuracyLabeled_parser += accuracyLabeled_parser
        devWords_parser += wordNum_parser


 #  newDevLoss_parser = devLoss_parser/devWords_parser
   newDevAccuracy_parser = float(devAccuracy_parser)/devWords_parser
   newDevAccuracyLabeled_parser = float(devAccuracyLabeled_parser)/devWords_parser
  # devLosses_parser.append(newDevLoss_parser)
   devAccuracies_parser.append(newDevAccuracy_parser)
   devAccuraciesLabeled_parser.append(newDevAccuracyLabeled_parser)

   return devLoss_parser/devWords_parser

while True:
#  corpus = getNextSentence("train")
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

       fromParser = doForwardPass(current)
       loss_parser, policy_related_loss_parser, _ ,_, wordNumInPass_parser = fromParser
#       print(fromLM, fromParser, fromDepL)

       if wordNumInPass_parser > 0:
         doBackwardPass(loss_parser, policy_related_loss_parser)
       else:
         print "No words, skipped backward"

       if counter % 50000 == 0:
          newDevLoss_parser = computeDevLoss()
          devLosses_parser.append(newDevLoss_parser)
          print "New dev loss Parser "+str(newDevLoss_parser)+". previous was: "+str(lastDevLoss_parser)

          if lastDevLoss_parser is None or newDevLoss_parser < lastDevLoss_parser:
             lastDevLoss_parser = newDevLoss_parser
             failedDevRuns_parser = 0
          else:
             failedDevRuns_parser += 1
             print "Skip Parser, hoping for better model"
             print devLosses_parser


          if failedDevRuns_parser > 0:
              quit()

#          if True:
#             print("DEMO, NOT SAVING SO FAR")
#             continue
          print "Saving"
          save_path = "/u/scr/mhahn/deps/"
          #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
          with open(save_path+"/manual_output_funchead_two_coarse_parser/"+args.language+"_"+__file__+"_model_"+str(myID)+".tsv", "w") as outFile:
             print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss_Parser", "AverageUAS", "AverageLAS", "DH_Weight","CoarseDependency","DistanceWeight", "EntropyWeight", "ObjectiveName"]))
             for i in range(len(itos_deps)):
                key = itos_deps[i]
                dhWeight = dhWeights[i].data.numpy()
                distanceWeight = distanceWeights[i].data.numpy()
                dependency = key
                print >> outFile, "\t".join(map(str,[myID, __file__, counter, devLosses_parser[-1], devAccuracies_parser[-1], devAccuraciesLabeled_parser[-1], dhWeight, dependency, distanceWeight, args.entropy_weight, objectiveName]))
       if args.maxNumberEpochs == len(devLosses_parser):
           quit()

#   devAccuracies_parser.append(newDevAccuracy_parser)
#   devAccuraciesLabeled_parser.append(newDevAccuracyLabeled_parser)



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
