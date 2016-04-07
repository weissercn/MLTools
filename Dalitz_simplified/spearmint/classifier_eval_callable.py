import sys
sys.path.insert(0,'../')
import classifier_eval_simplified

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return classifier_eval_simplified(params['aC'], params['agamma'])
