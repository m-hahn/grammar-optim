#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

from math import log, exp
from random import random, shuffle

from corpusIterator_FuncHead import CorpusIteratorFuncHead

languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]

with open("branching_entropy_coarse_byRelation.tsv", "w") as outFile:
 print >> outFile, "Language\tCoarseDependency\tBranchingEntropy"
 for language in languages:
    
    posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 
    
    posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]
    
    deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 
    
    #deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]
    
    
    
    header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]
    
    
    originalDistanceWeights = {}
    
    
    
    orderTable = {}
    keys = set()
    vocab = {}
    distanceSum = {}
    distanceCounts = {}
    depsVocab = set()
    totalCount = 0
    for partition in ["train", "dev"]:
      for sentence in CorpusIteratorFuncHead(language,partition, storeMorph=True).iterator():
       for line in sentence:
           vocab[line["word"]] = vocab.get(line["word"], 0) + 1
           line["coarse_dep"] = line["dep"][:(line["dep"]+":").index(":")]
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
           totalCount += 1
    #print orderTable
    entropyTotal = 0
    dhLogits = {}
    for key in keys:
       hd = orderTable.get((key, "HD"), 0) + 0.00000001
       dh = orderTable.get((key, "DH"), 0) + 0.00000001
       p_hd = hd/(hd+dh)
       entropyHere = p_hd * log(p_hd) + (1-p_hd) * log(1-p_hd)
   #    entropyTotal -= (hd+dh)/totalCount * entropyHere
       print >> outFile, ("\t".join(map(str,[language, key, -entropyHere])))
    
    
