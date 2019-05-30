#/u/nlp/bin/stake.py -g 11.5g -s run-stats-pretrain2.json "python readDataDistEnglishGPUFree.py"

import random
import sys

languages = ["English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Old_Church_Slavonic", "Ancient_Greek", "Hindi", "Swedish", "German", "Urdu"]
languages = sorted(list(set(languages)))
assert len(languages) == 51

sizes = []

with open("../results/corpus-size/corpus-sizes.tsv", "w") as outFile:
   print >> outFile, "\t".join(map(str, ["language", "sents_train", "sents_dev", "words_train", "words_dev"]))
   from corpusIterator_FuncHead import CorpusIteratorFuncHead
   for language in languages:
    sentsPerPart = {}
    wordsPerPart = {}
    for partition in ["train", "dev"]:
      sentsPerPart[partition] = 0
      wordsPerPart[partition] = 0
      corpus = CorpusIteratorFuncHead(language, partition=partition).iterator()
      for sentence in corpus:
       sentsPerPart[partition] += 1
       for line in sentence:
          if line["posUni"] != "PUNCT":
              wordsPerPart[partition] += 1
    print >> outFile, "\t".join(map(str, [language, sentsPerPart["train"], sentsPerPart["dev"], wordsPerPart["train"], wordsPerPart["dev"]]))
