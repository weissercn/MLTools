import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from sklearn.svm import SVC

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.0.0.txt",os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.200.1.txt")]
    comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.0_1.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.01_1.txt")]    
    clf = SVC(C=params['aC'],gamma=params['agamma'],probability=True, cache_size=7000)

    args=["dalitz","particle","antiparticle",100,comp_file_list,2,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13)]

    result= classifier_eval_simplified.classifier_eval(2,0,args)

    with open("optimisation_values.txt", "a") as myfile:
        myfile.write(str(params['aC'][0])+"\t"+ str(params['agamma'][0])+"\t"+str(result)+"\n")
    return result
