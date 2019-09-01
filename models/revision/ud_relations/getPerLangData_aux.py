#with open("../../../results/languages/languages.tsv", "r") as inFile:
#   languages = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[1:]]
#print(languages)

languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Ancient_Greek", "Old_Church_Slavonic"]
assert len(languages) == 51, len(languages)


from corpusIterator_FuncHead import CorpusIteratorFuncHead

counts = {}
directions = {}

counts_total = {}
directions_total = {}

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
          dep = line["dep"]
          if dep.startswith("aux"):
#              print(line["word"]+" "+line["lemma"])
              lemma = line["lemma"]+"_"+dep
              if "aux" not in counts:
                   counts["aux"] = {}
                   directions["aux"] = {}
                   counts_total["aux"] = {}
                   directions_total["aux"] = {}

              if language not in counts["aux"]:
                   counts["aux"][language] = {}
                   directions["aux"][language] = {}
                   counts_total["aux"][language] = 0
                   directions_total["aux"][language] = 0

              if lemma not in counts["aux"][language]:
                  counts["aux"][language][lemma] = 0
                  directions["aux"][language][lemma] = 0
              counts["aux"][language][lemma] += 1
              directions["aux"][language][lemma] += (1 if line["index"] > line["head"] else -1) # 1 == DH order
              counts_total["aux"][language] += 1
              directions_total["aux"][language] += (1 if line["index"] > line["head"] else -1) # 1 == DH order

with open("auxiliaries.tsv", "w") as outFile:
  for dep in sorted(list(counts)):
     coarse = dep[:dep.index(":")] if ":" in dep else dep
     for lang in sorted(list(counts[dep])):
       direction_total = directions_total[dep][lang]
       count_total = counts_total[dep][lang]
       P_total = (direction_total/float(count_total) + 1)/2
       assert P_total >= 0, P_total
       assert P_total <= 1, P_total

       for lemma in sorted(list(counts[dep][lang]), key=lambda x:-counts[dep][lang][x]):
         direction = directions[dep][lang][lemma]
         count = counts[dep][lang][lemma]
         if count < 10:
            continue
         # direction = count * (P - (1-P)) = count * (2P-1)
         P = (direction/float(count) + 1)/2
         assert P >= 0, P
         assert P <= 1, P
         
         print >> outFile, ("\t".join([str(x) for x in [coarse, dep, lang, lemma, count, P, P_total]]))

