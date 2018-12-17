import os
import random
import accessISWOCData
import accessTOROTData
import sys

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


from corpusIterator import CorpusIterator



 
def reverse_content_head(sentence):
   CH_CONVERSION_ORDER = ["cc", "case", "cop", "mark"]
   # find paths that should be reverted
   for dep in CH_CONVERSION_ORDER:
      for i in range(len(sentence)):
         if sentence[i]["dep"] == dep or sentence[i]["dep"].startswith(dep+":"):
             head = sentence[i]["head"]-1
             grandp = sentence[head]["head"]-1
             assert head > -1
             
             # grandp -> head -> i
             # grandp -> i -> head
             sentence[i]["head"] = grandp+1
             sentence[head]["head"] = i+1

             sentence[i]["dep"] = sentence[head]["dep"]
             sentence[head]["dep"] = "lifted_"+dep
             assert sentence[i]["index"] == i+1
#             print(i, dep)
 #            print(sentence)
  #           quit()
   return sentence

class CorpusIteratorFuncHead():
   def __init__(self, language, partition="train", storeMorph=False, splitLemmas=False):
      self.basis = CorpusIterator(language, partition=partition, storeMorph=storeMorph, splitLemmas=splitLemmas)
   def permute(self):
      self.basis.permute()
   def length(self):
      return self.basis.length()
   def iterator(self, rejectShortSentences = False):
     iterator = self.basis.iterator(rejectShortSentences=rejectShortSentences)
     for sentence in iterator:
         reverse_content_head(sentence)
         yield sentence
   def getSentence(self, index):
      return reverse_content_head(self.basis.getSentence(index))


class CorpusIteratorFuncHeadFraction():
   def __init__(self, language, partition="train", fraction=1.0, storeMorph=False, splitLemmas=False):
      self.basis = CorpusIterator(language, partition=partition, storeMorph=storeMorph, splitLemmas=splitLemmas, shuffleDataSeed=4)
      self.basis.data = self.basis.data[:int(fraction*len(self.basis.data))]
      self.permute()
      self.fraction = fraction
   def permute(self):
      self.basis.permute()
   def length(self):
      return self.basis.length()
   def iterator(self, rejectShortSentences = False):
     iterator = self.basis.iterator(rejectShortSentences=rejectShortSentences)
     counter = 0
     print("Actual length", self.length())
     for sentence in iterator:
#         if counter > self.fraction * self.length():
 #           break
  #          counter += 1
         reverse_content_head(sentence)
         yield sentence
   def getSentence(self, index):
      return reverse_content_head(self.basis.getSentence(index))


