import numpy as np
import math

def branin(aC, agamma):

    result = np.square(agamma - (5.1/(4*np.square(math.pi)))*np.square(aC) + 
         (5/math.pi)*aC - 6) + 10*(1-(1./(8*math.pi)))*np.cos(aC) + 10
    
    result = float(result)
    
    print 'Result = %f' % result
    #time.sleep(np.random.randint(60))
    return result

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return branin(params['aC'], params['agamma'])
