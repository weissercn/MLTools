import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import classifier_eval_simplified

def classifier_eval_callable(aC, agamma):

    a= classifier_eval_simplified.classifier_eval(aC, agamma,2)    
    return a

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    #return classifier_eval_callable(params['aC'], params['agamma'])
    return classifier_eval_simplified.classifier_eval(params['aC'], params['agamma'],2)

