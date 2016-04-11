import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import classifier_eval_simplified

def branin(x, y):

    a= classifier_eval_simplified.classifier_eval_callable(1, 0.1,2)    
    return a

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return branin(params['x'], params['y'])


