with open("../languages/languages-iso_codes.tsv", "r") as inFile:
   languages = [x[1:-1].split('","')[1:] for x in inFile.read().strip().split("\n")[1:]]
   languages = dict(list(zip([x[0] for x in languages], [x[1:] for x in languages])))

with open("corpora-table.tex", "w") as outFile:
 with open("corpus-sizes.tsv", "r") as inFile:
   header=next(inFile).strip().split("\t")
   header=dict(list(zip(header, list(range(len(header))))))
   for line in inFile:
      line = line.strip().split("\t")
      language = line[header["language"]]
      iso_code, family = languages[language]
      
      print(" & ".join([language.replace("_", " "), iso_code, family.replace("_", " "), line[header["sents_train"]]+"/"+line[header["sents_dev"]], line[header["words_train"]]+"/"+line[header["words_dev"]]])+"   \\\\", file=outFile)



