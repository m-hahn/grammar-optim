import subprocess
import random


val = {}
for probVPBranching in sorted([0.1, 0.2, 0.3, 0.4], key=lambda k: random.random()): # 0.1, 0.2, 0.3, 0.4, 
    val["probVPBranching"] = probVPBranching
    for probObj in sorted([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], key=lambda k: random.random()):
      val["probObj"] = probObj
      val["probNPBranching"] = 0.3
      for i in [1,2]:
         val["correlation_xcomp"] = True if i ==1 else False
         val["dlm_xcomp"] = True if i == 1 else False
         val["correlation_acl"] = True if i == 1 else False
         command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "estimateParseability_CKY.py"]
         for a, b in val.iteritems():
           command.append("--"+a+"="+str(b))
         subprocess.call(command)   
      
