import subprocess
import random

# Correlations, but Anti-DLM

val = {}
for probVPBranching in sorted([0.1, 0.2, 0.3, 0.4], key=lambda k: random.random()): # 0.1, 0.2, 0.3, 0.4, 
    val["probVPBranching"] = probVPBranching
    for probObj in sorted([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], key=lambda k: random.random()):
      val["probObj"] = probObj
      for probNPBranching in [0.0, 0.3]:
        val["probNPBranching"] = probNPBranching
        for i in [1]: #[1,2]: # correlation but no DLM
           val["correlation_xcomp"] = True
           val["dlm_xcomp"] = False
           val["correlation_acl"] = True
           command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "estimateParseability_CKY.py"]
           for a, b in val.iteritems():
             command.append("--"+a+"="+str(b))
           subprocess.call(command)   
        
