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
#assert rnn_dim in [100, 200, 300]
bilinearSize = int(sys.argv[17]) if not shallow else 2*rnn_dim #random.choice([100, 300, 500]) 
useMean = False #random.choice([True, False])
input_dropoutRate = float(sys.argv[18]) #random.choice([0.0, 0.1])
assert input_dropoutRate in [0.0, 0.05, 0.1, 0.2]
labelMLPDimension = int(sys.argv[19]) #random.choice([100, 200, 300, 400])
assert labelMLPDimension in [100, 200, 300], labelMLPDimension

# English RANDOM_BY_TYPE 0.002 0.9 0.9 0.3 1000 15 50 3 True 300 2 0.05 200 300 742002619

maxNumberOfUpdates = int(sys.argv[20]) if len(sys.argv) > 20 else 20000

model = sys.argv[21]
BASE_DIR = sys.argv[22]

dataFraction = float(sys.argv[23])
assert dataFraction <= 1.0

FILE_NAME = "sososo_coarse_parser_plane_partialData"

assert len(sys.argv) == 24


names = ["rnn_dim", "lr_lm", "beta1", "beta2", "dropout_rate", "tokensPerBatch", "clip_at", "pos_embedding_size", "lstm_layers", "bilinearSize", "useMean", "shallow", "clip_norm", "input_dropoutRate", "labelMLPDimension", "lr_policy", "momentum_policy", "entropy_weight"]
params = [rnn_dim, lr_lm, beta1, beta2, dropout_rate, tokensPerBatch, clip_at, pos_embedding_size, lstm_layers, bilinearSize, useMean, shallow, clip_norm, input_dropoutRate, labelMLPDimension, lr_policy, momentum_policy, entropy_weight]

if prescribedID is not None and prescribedID != "NONE":
  myID = int(prescribedID)
else:
  myID = random.randint(0,10000000)

#with open("/juicier/scr120/scr/mhahn/deps/LOG"+language+"_"+FILE_NAME+"_model_"+str(myID)+".txt", "w") as outFile:
#    print >> outFile, " ".join(sys.argv)



posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]


from math import log, exp, sqrt
from random import random, shuffle, randint
import os

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_FuncHead import CorpusIteratorFuncHead, CorpusIteratorFuncHeadFraction

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
   allGradients = gradients_from_the_left_sum #+ sum(line.get("children_decisions_logprobs",[]))

#   if "linearization_logprobability" in line:
#      allGradients += line["linearization_logprobability"] # the linearization of this element relative to its siblings affects everything starting at the start of the constituent, but nothing to the left of it
#   else:
#      assert line["coarse_dep"] == "root"
#

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
import numpy as np

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()
logsoftmaxLabels =  torch.nn.LogSoftmax(dim=2)



def orderChildrenRelative(sentence, remainingChildren, reverseSoftmax):
       childrenLinearized = []
       while len(remainingChildren) > 0:
           logits = [distanceWeights[stoi_deps[sentence[x-1]["dependency_key"]]] for x in remainingChildren]
           #print logits
           if reverseSoftmax:
              logits = [-x for x in logits]
           #print (reverseSoftmax, logits)
           softmax = logits #.view(1,-1).view(-1)
           selected = numpy.argmax(softmax)
           #selected = numpy.random.choice(range(0, len(remainingChildren)), p=softmax.data.numpy())
           #log_probability = torch.log(softmax[selected])
           assert "linearization_logprobability" not in sentence[remainingChildren[selected]-1]
           #sentence[remainingChildren[selected]-1]["linearization_logprobability"] = log_probability
           childrenLinearized.append(remainingChildren[selected])
           del remainingChildren[selected]
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
      probability = 1/(1 + exp(-dhLogit))
      dhSampled = (0.5 < probability)
#      logProbabilityGradient = (1 if dhSampled else -1) * (1-probability)
#      line["ordering_decision_gradient"] = logProbabilityGradient
      line["ordering_decision_log_probability"] = 0 #torch.log(1/(1 + torch.exp(- (1 if dhSampled else -1) * dhLogit)))

      
     
      direction = "DH" if dhSampled else "HD"
#torch.exp(line["ordering_decision_log_probability"]).data.numpy(),
      if printThings: 
         print "\t".join(map(str,["ORD", line["index"], (line["word"]+"           ")[:10], ("".join(list(key)) + "         ")[:22], line["head"], dhSampled, direction, (str(probability)+"      ")[:8], str(1/(1+exp(-dhLogits[key])))[:8], (str(distanceWeights[stoi_deps[key]])+"    ")[:8] , str(originalDistanceWeights[key])[:8]    ]  ))

      headIndex = line["head"]-1
      sentence[headIndex]["children_"+direction] = (sentence[headIndex].get("children_"+direction, []) + [line["index"]])
#      sentence[headIndex]["children_decisions_logprobs"] = (sentence[headIndex].get("children_decisions_logprobs", []) + [line["ordering_decision_log_probability"]])



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
for i, key in enumerate(itos_deps):

   # take from treebank, or randomize
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
    
   #print header
   #quit()
   # there might be a divergence because 'inferWeights...' models did not necessarily run on the full set of corpora per language (if there is no AllCorpora in the filename)
   #assert len(data) == len(itos_deps), [len(data), len(itos_deps)]
   if "Dependency" not in header:
      header[header.index("CoarseDependency")] = "Dependency"
   if "DH_Weight" not in header:
      header[header.index("DH_Mean_NoPunct")] = "DH_Weight"
   if "DistanceWeight" not in header:
      header[header.index("Distance_Mean_NoPunct")] = "DistanceWeight"

   for line in data:
   #   print line
#      head = line[header.index("Head")]
 #     dependent = line[header.index("Dependent")]
      dependency = line[header.index("Dependency")]
      key = dependency
      dhWeights[stoi_deps[key]] = temperature*float(line[header.index("DH_Weight")])
      distanceWeights[stoi_deps[key]] = temperature*float(line[header.index("DistanceWeight")])






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

vocab_size = 5

word_embedding_size = 0
# 0 EOS, 1 UNK, 2 BOS
#word_embeddings = torch.nn.Embedding(num_embeddings = vocab_size+3, embedding_dim = word_embedding_size).cuda()
pos_u_embeddings = torch.nn.Embedding(num_embeddings = len(posUni)+3, embedding_dim = pos_embedding_size).cuda()
pos_p_embeddings = torch.nn.Embedding(num_embeddings = len(posFine)+3, embedding_dim=pos_embedding_size).cuda()


#baseline = nn.Linear(50, 1).cuda()

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

#bilinearMatrix = nn.Bilinear(rnn_dim, rnn_dim, 1).cuda()

U = nn.Parameter(torch.Tensor(bilinearSize,bilinearSize).cuda())


biasHead = nn.Parameter(torch.Tensor(1,bilinearSize,1).cuda())
##biasDependent = nn.Parameter(torch.Tensor(bilinearSize).cuda())
#
#
#biasLabel = Variable(torch.Tensor((1+len(itos_pure_deps))).cuda(), requires_grad=True)
#U_label = Variable(torch.Tensor((1+len(itos_pure_deps)), bilinearSize,bilinearSize).cuda(), requires_grad=True)
#linear_Label_dep = Variable(torch.Tensor(bilinearSize, (1+len(itos_pure_deps))).cuda(), requires_grad=True)
#linear_Label_head = Variable(torch.Tensor(bilinearSize, (1+len(itos_pure_deps))).cuda(), requires_grad=True)
#


components = [ pos_u_embeddings, rnn, headRep, depRep, headMLPOut, dependentMLPOut, labelMLP, labelDecoder] # baseline, word_embeddings, pos_p_embeddings, 

def parameters_lm():
 for c in components:
   for param in c.parameters():
      yield param
 yield U
 yield biasHead
# yield biasLabel
# yield U_label
# yield linear_Label_dep
# yield linear_Label_head


#def parameters_ordering():
# yield dhWeights
# yield distanceWeights
#



from torch import optim

optimizer = optim.Adam(parameters_lm(), lr = lr_lm, betas=[beta1, beta2])
#optimizer = optim.SGD(parameters_lm(), lr = 0.0002, momentum=0.9) #, betas=[0.9, 0.9])
#optimizer_policy = optim.SGD(parameters_ordering(), lr = lr_policy, momentum=momentum_policy)


#for pa in parameters():
#  print pa

initrange = 0.01
#word_embeddings.weight.data.uniform_(-initrange, initrange)
pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
pos_p_embeddings.weight.data.uniform_(-initrange, initrange)


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


#
def prod(x):
   r = 1
   for s in x:
     r *= s
   return r

crossEntropy = 10.0

def encodeWord(w):
   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)


import torch.cuda
import torch.nn.functional


inputDropout = torch.nn.Dropout2d(p=input_dropoutRate)



content_pos = map(lambda x:stoi_pos_uni[x], filter(lambda y:y in itos_pos_uni, ["ADJ", "ADV", "NOUN", "NUM", "PROPN", "VERB"]))
function_pos = map(lambda x:stoi_pos_uni[x], filter(lambda y:y in itos_pos_uni,["ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT", "SCONJ", "SYM", "X"]))


#data = list(corpus)[:20]


baselinePerType = [4.0 for _ in itos_pure_deps]

def forward(current, computeAccuracy=False, doDropout=True):
       global biasHead
       global crossEntropy
       global batchSize
       batchSize = len(current)
#       if printHere:
#         print (batchSize, tokensPerBatch, sum(map(len, current)))
       batchOrderedLogits = zip(*map(lambda (y,x):orderSentence(x, dhLogits, y==0 and printHere), zip(range(len(current)),current)))
      
       batchOrdered = batchOrderedLogits[0]
       logits = batchOrderedLogits[1]
   
       lengths = map(len, batchOrdered)
       # current is already sorted by length
       maxLength = max(lengths)
#       if maxLength <= 2:
##         print current
#         print "Skipping extremely short sentence"
#         continue
       input_words = []
       input_pos_u = []
       input_pos_p = []
       for i in range(maxLength+2):
          input_words.append(map(lambda x: 2 if i == 0 else (encodeWord(x[i-1]["word"]) if i <= len(x) else 0), batchOrdered))
          input_pos_u.append(map(lambda x: 2 if i == 0 else (stoi_pos_uni[x[i-1]["posUni"]]+3 if i <= len(x) else 0), batchOrdered))
          input_pos_p.append(map(lambda x: 2 , batchOrdered)) # if i == 0 else (stoi_pos_ptb[x[i-1]["posFine"]]+3 if i <= len(x) else 0)

       hidden = None #(Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()), Variable(torch.FloatTensor().new(2, batchSize, rnn_dim).zero_()))
       loss = 0
       #wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0
#       for param in parameters_lm():
#         if param.grad is not None:
#           param.grad.data.fill_(0)
##       for c in components:
##          c.zero_grad()
#
##       momentum = 0.9
#       for p in  [dhWeights, distanceWeights]:
#          if p.grad is not None:
#             p.grad.data = p.grad.data.mul(momentum)
       optimizer.zero_grad()
       #optimizer_policy.zero_grad()


# (Variable(weight.new(2, bsz, self.nhid).zero_()),Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
#       for i in range(maxLength+1):
       if True:
           # TODO word dropout could also be added: randomly sprinkle `1' (UNK) in the LongTensor (not into input_words -- this will also be used for the softmax!)
#           words_layer = word_embeddings(Variable(torch.LongTensor(input_words)).cuda())
           pos_u_layer = pos_u_embeddings(Variable(torch.LongTensor(input_pos_u).transpose(0,1)).cuda())
#           pos_p_layer = pos_p_embeddings(Variable(torch.LongTensor(input_pos_p)).cuda())
           inputEmbeddings = pos_u_layer # torch.cat([pos_u_layer, pos_p_layer], dim=2) # words_layer, 
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
 #             if printHere:
 #               print inputEmbeddings
              inputEmbeddings = dropout(inputEmbeddings)
#           print hidden
#           print inputEmbeddings
           output, hidden = rnn(inputEmbeddings, hidden)

           outputFlat = output.contiguous().view(-1, 2*rnn_dim)
           if not shallow:
              heads = headRep(outputFlat)
              if doDropout:
                 heads = dropout(heads)
              heads = nn.ReLU()(heads)
#              if printHere:
#                print heads
#                print torch.nonzero(heads.data.view(-1)).size()
#                print prod(heads.size())
   #           heads = headMLPOut(heads)
              dependents = depRep(outputFlat)
              if doDropout:
                 dependents = dropout(dependents)
              dependents = nn.ReLU()(dependents)
#              if printHere:
#                print dependents
#                print torch.nonzero(dependents.data.view(-1)).size()
#                print prod(dependents.size())
           else:
              heads = outputFlat
              if doDropout:
                 heads = dropout(heads)
              dependents = outputFlat
              if doDropout:
                 dependents = dropout(dependents)


#           dependents = dependentMLPOut(dependents)

 #          baseline_predictions = baseline(words_layer.detach())

           
           heads = heads.view(batchSize, maxLength+2, 1, bilinearSize).contiguous() # .expand(batchSize, maxLength+2, maxLength+2, rnn_dim)
           dependents = dependents.view(batchSize, 1, maxLength+2, bilinearSize).contiguous() # .expand(batchSize, maxLength+2, maxLength+2, rnn_dim)
#           if printHere:
#              print "HEADS"
#              print heads
#              print dependents
#           dependents = dependents.view(-1, rnn_dim)

           
           part1 = torch.matmul(heads, U)
#           print part1.size()
#           print torch.transpose(dependents, 2, 3).size() # torch.transpose(dependents, 1, 2).size()
           bilinearAttention = torch.matmul(part1, torch.transpose(dependents, 2, 3)) # 

           #biasHead = biasHead.view(1,bilinearSize,1) #.unsqueeze(0).unsqueeze(0)
           heads = heads.view(-1, 1, bilinearSize)
         
#           print biasHead.size()
#           print heads.size()
#           print bilinearAttention.size()
#           print torch.matmul(heads,biasHead).view(batchSize, 1, 1, maxLength+2).size()

           biasFromHeads = torch.matmul(heads,biasHead).view(batchSize, 1, 1, maxLength+2)
#           if printHere:
#              print biasFromHeads
           bilinearAttention = bilinearAttention + biasFromHeads




#           print bilinearAttention.size()
#           quit()

#           bilinearAttention = bilinearMatrix(heads, dependents)
#           print bilinearAttention
           bilinearAttention = bilinearAttention.view(-1, maxLength+2)
 #          print bilinearAttention
#           if printHere:
#             print "BILINEAR"
#             print bilinearAttention


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
#               posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 
               if False and pos-3 in function_pos:
                  continue
               else:
                  assert True or pos-3 in content_pos, (pos-3, itos_pos_uni[pos-3]) 
               if batchOrdered[j][i-1]["head"] == 0:
                  realHead = 0
               else:
                  realHead = batchOrdered[j][i-1]["reordered_head"] # this starts at 1, so just right for the purposes here
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
           #print heads.size()
           #print targetIndices.size()
           headStates = torch.gather(heads, 1, targetIndices)
           #print headStates
           dependents = dependents.view(batchSize, maxLength+2, bilinearSize)
           headStates = headStates.view(batchSize, maxLength+2, bilinearSize)
           headsAndDependents = torch.cat([dependents, headStates], dim=2)
           labelHidden = labelMLP(headsAndDependents)
           if doDropout:
              labelHidden = dropout(labelHidden)
           labelHidden = nn.ReLU()(labelHidden)
           labelLogits = labelDecoder(labelHidden)
           labelSoftmax = logsoftmaxLabels(labelLogits)

   #        print labelSoftmax
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
#           policyGradientLosses = [0 for _ in range(batchSize)]
         #  policyGradientLoss = 0
        #   for j in range(batchSize):
       #      lossForThisSentenceMinusBaselines = 0
      #       for i in range(1,len(batchOrdered[j])+1):
     #          pos = input_pos_u[i][j]
    #           assert pos >= 3, (i,j)
   #            if False and pos-3 in function_pos:
  #                continue
 #              else:
#                  assert True or pos-3 in content_pos  
              # lossForThisSentenceMinusBaselines += (lossesHeadsAndLabels[j][i]  - log(len(batchOrdered[j])) - baselinePerType[stoi_pure_deps[batchOrdered[j][i-1]["dep"]]])
              # if printHere:
              #    print (lossesHeadsAndLabels[j][i], baselinePerType[stoi_pure_deps[batchOrdered[j][i-1]["dep"]]] + log(len(batchOrdered[j])), (lossesHeadsAndLabels[j][i]  - log(len(batchOrdered[j])) - baselinePerType[stoi_pure_deps[batchOrdered[j][i-1]["dep"]]]))
#             print batchOrdered[j][-1]["relevant_logprob_sum"] * lossForThisSentenceMinusBaselines
#             policyGradientLoss += batchOrdered[j][-1]["relevant_logprob_sum"] * lossForThisSentenceMinusBaselines

       #    for j in range(batchSize):
        #     for i in range(1,len(batchOrdered[j])+1):
         #      baselinePerType[stoi_pure_deps[batchOrdered[j][i-1]["dep"]]] = 0.99 * baselinePerType[stoi_pure_deps[batchOrdered[j][i-1]["dep"]]] + (1-0.99) * (lossesHeadsAndLabels[j][i] - log(len(batchOrdered[j])))



           if computeAccuracy:
              wordNumAcc = 0
              for j in range(batchSize):
                  for i in range(1,len(batchOrdered[j])+1):
    #             for i in range(1,maxLength+1):
                       pos = input_pos_u[i][j]
                       assert pos >= 3, (i,j)
        #               posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 
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
                       if (True or input_words[i] > 2) and printHere and j == batchSize/2: 
#                          print predictions.data.cpu().numpy()
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
#       if printHere:
#         print "BACKWARD 1"
#       if printHere:
#         print "BACKWARD 2"

 #      probabilities = torch.sigmoid(dhWeights)
#       print ["MEAN PROBABILITIES", torch.mean(probabilities)]
       #print ["PG", policyGradientLoss]

#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

       policy_related_loss = 0 #entropy_weight * neg_entropy + policyGradientLoss # lives on CPU



       return loss, policy_related_loss, accuracy if computeAccuracy else None, accuracyLabeled if computeAccuracy else None, wordNum

def backward(loss, policy_related_loss):
       if loss is 0:
          print "Absolutely Zero Loss"
          print current
          return

       loss.backward()

#       policy_related_loss.backward()
#       if printHere:
#         print "BACKWARD 3 "+FILE_NAME+" "+language+" "+str(myID)+" "+str(counter)
#         print zip(names, params)
       torch.nn.utils.clip_grad_norm(parameters_lm(), clip_at, norm_type=clip_norm)
#       torch.nn.utils.clip_grad_norm(parameters_ordering(), 5.0, norm_type='inf')
       optimizer.step()
       #optimizer_policy.step()


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
         corpusDev = CorpusIteratorFuncHeadFraction(language, "dev", 0.2).iterator(rejectShortSentences = True)
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
#              print (loss.data.cpu().numpy()    , accuracy)
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
  corpus = CorpusIteratorFuncHeadFraction(language, "train", dataFraction).iterator(rejectShortSentences = True)
  partitions = getPartitions(corpus)
  epochs += 1
  for partition in partitions:
       if counter > maxNumberOfUpdates:
            print "Ran for a long time, quitting."
            quit()

#       if counter > 100:
#           break
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


#       if crossEntropy > 30:
#           with open("/juicier/scr120/scr/mhahn/deps/parsing-upos/performance-"+language+"_"+FILE_NAME+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
#              print >> outFile, " ".join(names)
#              print >> outFile, " ".join(map(str,params))
#              print >> outFile, " ".join(map(str,[100]))
#              print >> outFile, " ".join(map(str,[-1]))
#              print >> outFile, " ".join(map(str,[-1]))
#              print >> outFile, " ".join(sys.argv)
#           print "Loss exploding"
#           quit()

  if True: #counter % 5000 == 0:
         print >> sys.stderr, (myID, "EPOCHS", epochs, "UPDATES", counter)

         computeDevLoss()

         with open("/juicier/scr120/scr/mhahn/deps/parsing-upos/performance-"+language+"_"+FILE_NAME+"_model_"+str(myID)+"_"+model+".txt", "w") as outFile:
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

#         print "Saving"
#         save_path = "/juicier/scr120/scr/mhahn/deps/"
#         #save_path = "/afs/cs.stanford.edu/u/mhahn/scr/deps/"
#         with open(save_path+"/manual_output_funchead/"+language+"_"+FILE_NAME+"_model_"+str(myID)+".tsv", "w") as outFile:
#            print >> outFile, "\t".join(map(str,["FileName","ModelName","Counter", "AverageLoss","Head","DH_Weight","Dependency","Dependent","DistanceWeight", "EntropyWeight", "ObjectiveName"]))
#            for i in range(len(itos_deps)):
#               key = itos_deps[i]
#               dhWeight = dhWeights[i].data.numpy()
#               distanceWeight = distanceWeights[i].data.numpy()
#               head, dependency, dependent = key
#               print >> outFile, "\t".join(map(str,[myID, FILE_NAME, counter, devLosses[-1], head, dhWeight, dependency, dependent, distanceWeight, entropy_weight, objectiveName]))
#











 
