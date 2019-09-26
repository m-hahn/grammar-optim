languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]

assert len(languages) == 51
import os
import random


import subprocess

header = ["Language", "UnigramEntropy","SentenceCount","MedianArity","MedianTreeDepth","MeanArity","MeanTreeDepth","MaxArity","MaxTreeDepth","MedianSentenceLength","MeanSentenceLength","MaxSentenceLength"]


with open("corpusProperties.tsv", "w") as outFile:
  print >> outFile, "\t".join(header)
  for language in languages:
    data = [x.split("\t") for x in open("results/properties-"+language, "r").read().strip().split("\n")]
    assert len(data)+1 == len(header)
    values = [language] + [x[1] for x in data]
    print >> outFile, "\t".join(values)         

