import subprocess
import random

val = {}
for _ in range(100):
    val["probVPBranching"] = random.uniform(0.2, 0.4)
    val["probObj"] = random.uniform(0.0, 1.0)
    val["probNPBranching"] = 0.3
    for x in [True, False]:    
       val["correlation_xcomp"] = x #random.choice([True, False])
       for y in [True, False]:
          if x == False and y == False: # DLM only makes a difference if they are on the same side
             continue
          val["dlm_xcomp"] = y #random.choice([True, False])
          for z in [True, False]:
             val["correlation_acl"] = z #random.choice([True, False])
             command = ["/u/nlp/anaconda/ubuntu_16/envs/py27-mhahn/bin/python2.7", "estimateParseability_CKY.py"]
             for a, b in val.iteritems():
               command.append("--"+a+"="+str(b))
             subprocess.call(command)   
    
