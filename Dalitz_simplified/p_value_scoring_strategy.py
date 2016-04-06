from __future__ import print_function

import numpy as np
from scipy import stats


def p_value_getter(y,y_pred,prob_pred):
	#making sure the inputs are row vectors
	y         = np.reshape(y,(1,y.shape[0]))
	y_pred    = np.reshape(y_pred,(1,y_pred.shape[0]))
	prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

	prob_0    = prob_pred[np.logical_or.reduce([y==0])]
	prob_1    = prob_pred[np.logical_or.reduce([y==1])]
	if __debug__:
		print("Plot")
	p_KS=stats.ks_2samp(prob_0,prob_1)
	print(p_KS)
	return p_KS

if __name__ == "__main__":
	#This code is only executed when this function is called directly, not when it is imported
	print("Testing the p_value_getter function, which can be imported using 'import p_value_getter' and used by typing 'p_KS=p_value_getter.p_value_getter(y,y_pred,prob_pred)'")
	y= np.array([1,0,0])
	y_pred=np.array([1,0,1])
	p_pred=np.array([0.8,0.2,0.6])
	p_value_getter(y,y_pred,p_pred)



