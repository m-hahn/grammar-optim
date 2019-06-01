
import sys
if len(sys.argv) > 1:
   languages = sys.argv[1].split(",")
else:
   languages = ["Hindi", "Swedish", "German", "Urdu", "English", "Spanish", "Chinese", "Slovenian", "Estonian", "Norwegian", "Serbian", "Croatian", "Finnish", "Portuguese", "Catalan", "Russian", "Arabic", "Czech", "Japanese", "French", "Latvian", "Basque", "Danish", "Dutch", "Ukrainian", "Hebrew", "Hungarian", "Persian", "Bulgarian", "Romanian", "Indonesian", "Greek", "Turkish", "Slovak", "Belarusian", "Galician", "Italian", "Lithuanian", "Polish", "Vietnamese", "Korean", "Tamil", "Irish", "Marathi", "Afrikaans", "Telugu" , "Coptic", "Gothic",  "Latin", "Ancient_Greek", "Old_Church_Slavonic"]

import random
import subprocess
import os

while True:
   script = "readDataDistCrossGPUFreeAllTwoEqual_NoClip_ByCoarseOnly_FixObj_OnlyLangmod_Replication.py"
   args = {}

   relevantLanguages = [language for language in languages if len([x for x in os.listdir("../../../../raw-results/manual_output_funchead_langmod_coarse/") if x.startswith(language+"_")]) < 8]
   if len(relevantLanguages) == 0:
     quit()
   args["language"] = random.choice(relevantLanguages)

   args["entropy_weight"] = random.choice([0.0001, 0.0001, 0.0001,0.001, 0.001,  0.001, 0.001,0.001, 0.001]) 
   args["lr_policy"] = random.choice([0.000005, 0.00001, 0.00001, 0.00001, 0.00002, 0.00002, 0.00003]) 
   args["momentum_policy"] = random.choice([0.8, 0.9])
   args["lr_baseline_lm"] = random.choice([1.0])
   args["dropout_prob_lm"] = random.choice([0.0, 0.3, 0.5])
   args["lr_lm"] = random.choice([0.05, 0.1, 0.1, 0.1, 0.2])
   args["batchSize"] = random.choice([1])
 
   args["lr_parser"] = 0.001
   args["beta1"] =  0.9
   args["beta2"] = 0.999
   args["dropout_rate_parser"] = 0.2
   args["clip_at_parser"] = 15.0
   args["clip_norm_parser"] = 2
   args["pos_embedding_size_parser"] = 100
   args["lstm_layers_parser"] = 2
   args["shallow"] = True
   args["rnn_dim_parser"] = 200
   args["bilinearSize"] = 300
   args["input_dropoutRate_parser"] = 0.0
   args["labelMLPDimension"] = 300

   
  
   command = map(str,["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", script] +(" ".join(["--"+x+" "+str(y) for x,y in args.iteritems() ])).split(" ")   )
   print(command)
   subprocess.call(command)
   #break 
