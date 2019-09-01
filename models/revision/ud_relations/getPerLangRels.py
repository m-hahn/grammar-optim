#with open("../../../results/languages/languages.tsv", "r") as inFile:
#   languages = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[1:]]
#print(languages)

languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
assert len(languages) == 51, len(languages)


from corpusIterator_FuncHead import CorpusIteratorFuncHead

counts = {}
directions = {}

for language in languages: #[:2]:
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
          dep = line["dep"]
          if dep not in counts:
               counts[dep] = {}
               directions[dep] = {}
          if language not in counts[dep]:
               counts[dep][language] = 0
               directions[dep][language] = 0
          counts[dep][language] += 1
          directions[dep][language] += (1 if line["index"] > line["head"] else -1) # 1 == DH order
with open("relations.tsv", "w") as outFile:
  for dep in sorted(list(counts)):
     coarse = dep[:dep.index(":")] if ":" in dep else dep
     for lang in sorted(list(counts[dep])):
         direction = directions[dep][lang]
         count = counts[dep][lang]
         # direction = count * (P - (1-P)) = count * (2P-1)
         P = (direction/float(count) + 1)/2
         assert P >= 0, P
         assert P <= 1, P
         
         print >> outFile, ("\t".join([str(x) for x in [coarse, dep, lang, counts[dep][lang], P]]))

