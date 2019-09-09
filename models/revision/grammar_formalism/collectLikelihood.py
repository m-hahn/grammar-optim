languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Gothic", "Hebrew", "Hungarian", "Latin", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu", "Coptic", "Old_Church_Slavonic", "Ancient_Greek"]

import random
random.shuffle(languages)
import os
import subprocess
types = ["computeLikelihood.py", "computeLikelihood_PureUD.py"]
with open("likelihood-results.tsv", "w") as outFile:
    print >> outFile, "\t".join(["Language"] + [x.replace(".py","") for x in types])
    for language in languages:
       r = {}
       for typ in types:
          if language+"_"+typ+".txt" in os.listdir("../../../raw-results/treebank_likelihood/"):
             result = float(open("../../../raw-results/treebank_likelihood/"+language+"_"+typ+".txt", "r").read().strip())
             r[typ] = result
          else:
             r[typ] = "NA"
       print >> outFile, "\t".join([str(x) for x in [language] + [r[typ] for typ in types]])
    
     
