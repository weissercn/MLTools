from __future__ import print_function

import numpy as np
from scipy import stats


def p_value_scoring_object(clf, X, y):
	"""
	p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
	"""

	#Finding out the prediction probabilities
	prob_pred=clf.predict_proba(X)[:,1]
	#print(prob_pred)

	#making sure the inputs are row vectors
	y         = np.reshape(y,(1,y.shape[0]))
	prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

	#Separate prob into particle and antiparticle samples
	prob_0    = prob_pred[np.logical_or.reduce([y==0])]
	prob_1    = prob_pred[np.logical_or.reduce([y==1])]
	#if __debug__:
		#print("Plot")
	p_KS_stat=stats.ks_2samp(prob_0,prob_1)
	print(p_KS_stat)
	p_KS=-p_KS_stat[1]
	return p_KS

if __name__ == "__main__":
	#This code is only executed when this function is called directly, not when it is imported
	print("Testing the p_value_getter function, which can be imported using 'import p_value_scoring_object' and used by typing 'p_KS=p_value_scoring_object.p_value_scoring_object(clf, X, y)'")
	from sklearn.datasets import load_iris
	from sklearn.svm import SVC
	#Load data set
	iris = load_iris()
	X = iris.data
	y = iris.target
	#To make sure there are only two classes and then shuffling
	data=np.c_[X[:100,:],y[:100]]
	np.random.shuffle(data)
	X = data[:,:-1]
	y = data[:,-1]	
	#print(X.shape)
	#print(y)
	clf= SVC(probability=True)
	clf.fit(X[:50,:],y[:50])
	print(p_value_scoring_object(clf,X[50:,:],y[50:]))



