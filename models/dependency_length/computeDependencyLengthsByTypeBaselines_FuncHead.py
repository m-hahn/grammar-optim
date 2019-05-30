import os
import subprocess

path = "/u/scr/corpora/Universal_Dependencies_2.1/ud-treebanks-v2.1/"
files = os.listdir(path)
languages = list(set(map(lambda name:name[name.index("_")+1:].split("-")[0], filter(lambda name:name.startswith("UD_"), files))))

for language in languages:
   print language
   subprocess.call(["python", "computeDependencyLengths/computeDependencyLengthsRandomProjectiveByType_FuncHead.py", language, language, "RANDOM"])
   subprocess.call(["python", "computeDependencyLengths/computeDependencyLengthsRealByType_FuncHead.py", language, language])

