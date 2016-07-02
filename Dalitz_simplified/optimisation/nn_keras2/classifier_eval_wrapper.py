import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval_simplified
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.utils import np_utils, generic_utils
from keras.wrappers.scikit_learn import KerasClassifier

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.0.0.txt",os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.200.1.txt")]
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.0_1.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.01_1.txt")]  
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/accept_reject/legendre_data/data_sin1diff_5_and_5_periods10D_1000points_optimisation_sample_0.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/accept_reject/legendre_data/data_sin1diff_5_and_6_periods10D_1000points_optimisation_sample_0.txt")]
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_4D_1000_0.6_0.2_0.1_optimisation_0.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_4D_1000_0.6_0.2_0.075_optimisation_0.txt")]

    #clf = "This should not be used as Keras mode is turned on" 
    clf = KerasClassifier(classifier_eval_simplified.make_keras_model,n_hidden_layers=params['n_hidden_layers'],dimof_middle=params['dimof_middle'],dimof_input=2)


    args=["Dalitz_keras2","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13),0]

    result= classifier_eval_simplified.classifier_eval(2,2,args)

    with open("optimisation_values.txt", "a") as myfile:
        myfile.write(str(params['dimof_middle'][0])+"\t"+ str(params['n_hidden_layers'][0])+"\t"+str(result)+"\n")
    return result
