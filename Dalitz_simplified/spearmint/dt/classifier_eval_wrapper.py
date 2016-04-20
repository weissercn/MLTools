import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn import tree

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.0.0.txt",os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.200.1.txt")]
    
    clf = tree.DecisionTreeClassifier('gini','best',params['max_depth'], params['min_samples_split'], 1, 0.0, None) 

    args=["dalitz","particle","antiparticle",100,comp_file_list,2,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13)]

    result= classifier_eval_simplified.classifier_eval(2,0,args)

    with open("optimisation_values.txt", "a") as myfile:
        myfile.write(str(params['max_depth'][0])+"\t"+ str(params['min_samples_split'][0])+"\t"+str(result)+"\n")
    return result
