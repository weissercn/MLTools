import numpy as np
import math
import sys 
sys.path.insert(0,'../')
import classifier_eval_simplified

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    result= classifier_eval_simplified.classifier_eval(params['aC'], params['agamma'],2)
    with open("optimisation_values.txt", "a") as myfile:
    	myfile.write(str(params['aC'])+"\n"+ str(params['agamma'])+"\n"+str(result))
    return result
