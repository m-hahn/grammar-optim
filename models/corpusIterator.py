import os
import random
import accessISWOCData
import accessTOROTData
import sys

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


def readUDCorpus(language, partition):
      basePaths = ["/u/scr/corpora/Universal_Dependencies_2.2/ud-treebanks-v2.2/", "/u/scr/corpora/Universal_Dependencies_2.1/ud-treebanks-v2.1/"]
      files = []
      while len(files) == 0:
        if len(basePaths) == 0:
           print "No files found"
           raise IOError
        basePath = basePaths[0]
        del basePaths[0]
        files = os.listdir(basePath)
        files = filter(lambda x:x.startswith("UD_"+language.replace("-Adap", "")), files)
      data = []
      for name in files:
        if "Sign" in name:
           print "Skipping "+name
           continue
        assert ("Sign" not in name)
        if "Chinese-CFL" in name:
           print "Skipping "+name
           continue
        suffix = name[len("UD_"+language):]
        if name == "UD_French-FTB":
            subDirectory = "/juicier/scr120/scr/mhahn/corpus-temp/UD_French-FTB/"
        else:
            subDirectory =basePath+"/"+name
        subDirFiles = os.listdir(subDirectory)
        partitionHere = partition
        if (name in ["UD_North_Sami", "UD_Irish", "UD_Buryat-BDT", "UD_Armenian-ArmTDP"]) and partition == "dev" and (not language.endswith("-Adap")):
            print "Substituted test for dev partition"
            partitionHere = "test"
        elif language.endswith("-Adap"):
          if (name in ["UD_Kazakh-KTB", "UD_Cantonese-HK", "UD_Naija-NSC", "UD_Buryat-BDT", "UD_Thai-PUD", "UD_Breton-KEB", "UD_Faroese-OFT", "UD_Amharic-ATT", "UD_Kurmanji-MG", "UD_Upper_Sorbian-UFAL"]):
             partitionHere = "test"
          elif name == "UD_Armenian-ArmTDP":
             partitionHere  = ("train" if partition == "dev" else "test")
            
        candidates = filter(lambda x:"-ud-"+partitionHere+"." in x and x.endswith(".conllu"), subDirFiles)
        if len(candidates) == 0:
           print "Did not find "+partitionHere+" file in "+subDirectory
           continue
        if len(candidates) == 2:
           candidates = filter(lambda x:"merged" in x, candidates)
        assert len(candidates) == 1, candidates
        try:
           dataPath = subDirectory+"/"+candidates[0]
           with open(dataPath, "r") as inFile:
              newData = inFile.read().strip().split("\n\n")
              assert len(newData) > 1
              if language.endswith("-Adap")  and (name in ["UD_Kazakh-KTB", "UD_Cantonese-HK", "UD_Naija-NSC", "UD_Buryat-BDT",  "UD_Thai-PUD", "UD_Breton-KEB", "UD_Faroese-OFT", "UD_Amharic-ATT", "UD_Kurmanji-MG", "UD_Upper_Sorbian-UFAL"]): # "UD_Armenian-ArmTDP",
                  random.Random(4).shuffle(newData)
                  devLength = 100
                  if partition == "dev":
                       newData = newData[:devLength]
                  elif partition == "train":
                       newData = newData[devLength:]
                  else:
                       assert False
              data = data + newData
        except IOError:
           print "Did not find "+dataPath

      assert len(data) > 0, (language, partition, files)


      print >> sys.stderr, "Read "+str(len(data))+ " sentences from "+str(len(files))+" "+partition+" datasets."
      return data

class CorpusIterator():
   def __init__(self, language, partition="train", storeMorph=False, splitLemmas=False, shuffleData=True, shuffleDataSeed=None, splitWords=False):
      if splitLemmas:
           assert language == "Korean"
      self.splitLemmas = splitLemmas
      self.splitWords = splitWords
      assert self.splitWords == (language == "BKTreebank_Vietnamese")

      self.storeMorph = storeMorph
      if language.startswith("ISWOC_"):
          data = accessISWOCData.readISWOCCorpus(language.replace("ISWOC_",""), partition)
      elif language.startswith("TOROT_"):
          data = accessTOROTData.readTOROTCorpus(language.replace("TOROT_",""), partition)
      elif language == "BKTreebank_Vietnamese":
          import accessBKTreebank
          data = accessBKTreebank.readBKTreebank(partition)
      elif language == "TuebaJS":
         import accessTuebaJS
         data = accessTuebaJS.readTuebaJSTreebank(partition)
         assert len(data) > 0, (language, partition)
      elif language == "LDC2012T05":
         import accessChineseDependencyTreebank
         data = accessChineseDependencyTreebank.readChineseDependencyTreebank(partition)
         assert len(data) > 0, (language, partition)
        
      else:
          data = readUDCorpus(language, partition)
      if shuffleData:
       if shuffleDataSeed is None:
         random.shuffle(data)
       else:
         random.Random(shuffleDataSeed).shuffle(data)

      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def processSentence(self, sentence):
        sentence = map(lambda x:x.split("\t"), sentence.split("\n"))
        result = []
        for i in range(len(sentence)):
#           print sentence[i]
           if sentence[i][0].startswith("#"):
              continue
           if "-" in sentence[i][0]: # if it is NUM-NUM
              continue
           if "." in sentence[i][0]:
              continue
           sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(header)])
           sentence[i]["head"] = int(sentence[i]["head"])
           sentence[i]["index"] = int(sentence[i]["index"])
           sentence[i]["word"] = sentence[i]["word"].lower()
           if self.language == "Thai-Adap":
              assert sentence[i]["lemma"] == "_"
              sentence[i]["lemma"] = sentence[i]["word"]
           if "ISWOC" in self.language or "TOROT" in self.language:
              if sentence[i]["head"] == 0:
                  sentence[i]["dep"] = "root"

           if self.splitLemmas:
              sentence[i]["lemmas"] = sentence[i]["lemma"].split("+")

           if self.storeMorph:
              sentence[i]["morph"] = sentence[i]["morph"].split("|")

           if self.splitWords:
              sentence[i]["words"] = sentence[i]["word"].split("_")


           sentence[i]["dep"] = sentence[i]["dep"].lower()
           if self.language == "LDC2012T05" and sentence[i]["dep"] == "hed":
              sentence[i]["dep"] = "root"
           if self.language == "LDC2012T05" and sentence[i]["dep"] == "wp":
              sentence[i]["dep"] = "punct"

           result.append(sentence[i])
 #          print sentence[i]
        return result
   def getSentence(self, index):
      result = self.processSentence(self.data[index])
      return result
   def iterator(self, rejectShortSentences = False):
     for sentence in self.data:
        if len(sentence) < 3 and rejectShortSentences:
           continue
        yield self.processSentence(sentence)


