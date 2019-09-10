import random
 

itos_labels = []
stoi_labels = {}

itos = ["n", "v", "p", "c", "."]


d=0.2
m=0.5
r=0.5
s=0.0 # 0.8 English, 0.0 for German. Can use 0.2 instead to match Entropy with English?
# It seems that log_beta == 2.0 or 3.0 broadly produces the effects.

def sample(nt, extHead, dep): # samples a constituent, and returns the head
   if nt == "vp":
      p = random.random()
      if p < 0.5:
         head = {"posUni" : "v", "headPointer" : extHead, "dep" : dep}
         return [head], head
      elif p < 0.7:
         vp, v = sample("vp", extHead, dep)
         np, _ = sample("np", v, "obj")
         return vp+np, v
      else:
         vp, v = sample("vp", extHead, dep)
         compp, _ = sample("vp", v, "xcomp")
         return vp + compp, v
   elif nt == "np":
      p = random.random()
      if p < 0.8:
         head = {"posUni" : "n", "headPointer" : extHead, "dep" : dep}
         return [head], head
      else:
         np, n = sample("np", extHead, dep)
         vp, _ = sample("vp", n, "acl")
         return np+vp, n

def processIndices(x):
   for i in range(len(x)):
     x[i]["index"] = i+1
   for y in x:
     y["word"] = y["posUni"]
     y["posFine"] = y["posUni"]

     if y["headPointer"] is None:
         y["head"] = 0
         assert y["dep"] == "root"
     else:
         assert y["dep"] != "root"
         y["head"] = y["headPointer"]["index"]
         assert y is not y["headPointer"]
         assert y["head"] != y["index"], y
#   print(x)
   return x

def load(language, partition="train", removeMarkup=True, tokenize=True):
  for _ in range(30000 if partition == "train" else 10000):
     v = processIndices(sample("vp", None, "root")[0])
     if len(v) == 1:
       continue
     yield v

def training(language=None, removeMarkup=True, tokenize=True):
  return load(language, partition = 'train', removeMarkup=removeMarkup, tokenize=tokenize)

def dev(language=None, removeMarkup=True, tokenize=True):
  return load(language, partition = 'dev', removeMarkup=removeMarkup, tokenize=tokenize)

def addRegions(l):
   return ["_f"+x for x in l]

def test(language, removeMarkup=True, tokenize=True):
  chunk = []
  regions = []
  for _ in range(1000):
     chunk += list("ncncnvvv.")
     regions += ["n0", "c0", "n1", "c1", "n2", "v3", "v2", "v1g", ".g"]
     for _ in range(10): # ADD FILLERS
        l = sample("s")
        chunk += l
        regions += addRegions(l)
     chunk += list("ncncnvv.")
     regions += ["n0", "c0", "n1", "c1", "n2", "v3", "v1u", ".u"]
     for _ in range(10): # ADD FILLERS
        l = sample("s")
        chunk += l
        regions += addRegions(l)
  yield chunk, regions




if __name__ == '__main__':
    stream = test("dutch")
    print(next(stream) )
