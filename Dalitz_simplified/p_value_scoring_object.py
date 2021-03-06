from __future__ import print_function

import sys
import numpy as np
from scipy import stats


def p_value_scoring_object(clf, X, y):
	"""
	p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
	"""

	#Finding out the prediction probabilities
	prob_pred=clf.predict_proba(X)[:,1]
	#print(prob_pred)

	#This can be deleted if not using Keras
	#For Keras turn cathegorical y back to normal y
	if y.ndim==2:
		if y.shape[0]!=1 and y.shape[1]!=1:
			#Then we have a cathegorical vector
			y = y[:,1]

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

def p_value_scoring_object_visualisation(clf, X, y): 
        """ 
        p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
        """
	
        #Finding out the prediction probabilities
        prob_pred=clf.predict_proba(X)[:,1]
        #print(prob_pred)

        #This can be deleted if not using Keras
        #For Keras turn cathegorical y back to normal y
        if y.ndim==2:
                if y.shape[0]!=1 and y.shape[1]!=1:
                        #Then we have a cathegorical vector
                        y = y[:,1]

        #making sure the inputs are row vectors
        y         = np.reshape(y,(1,y.shape[0]))
        prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

        #Separate prob into particle and antiparticle samples
        prob_0    = prob_pred[np.logical_or.reduce([y==0])]
        prob_1    = prob_pred[np.logical_or.reduce([y==1])]
        #if __debug__:
                #print("Plot")
        p_KS_stat=stats.ks_2samp(prob_0,prob_1)

	#http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#example-tree-plot-iris-py
	import matplotlib.pyplot as plt
	n_classes =2 
        plot_colors = "br"

	if X.shape[1]!=2:
		print("The visualisation mode has only been implemented for 2 dimensions.")
		sys.exit(1)	

	#print("X[:,0].min() , ", X[:,0].min(), "X[:,0].max() : ", X[:,0].max())
	#print("X[:,0].min()*0.9 , ", X[:,0].min()*0.9, "X[:,0].max()*1.1 : ", X[:,0].max()*1.1)
	x_min, x_max = X[:, 0].min()*0.9 , X[:, 0].max() *1.1
    	y_min, y_max = X[:, 1].min() * 0.9, X[:, 1].max() * 1.1
	x_plot_step = (x_max - x_min)/20.0
	y_plot_step = (y_max - y_min)/20.0
	print("x_min : ", x_min, "x_max : ", x_max, "x_plot_step : ", x_plot_step)
	print("y_min : ", y_min, "y_max : ", y_max, "y_plot_step : ", y_plot_step)
	x_list=np.arange(x_min, x_max, x_plot_step)
	#print("x_list : ",x_list)
	y_list=np.arange(y_min, y_max, y_plot_step)
	#print("y_list : ", y_list)
    	xx, yy = np.meshgrid(x_list, y_list)
	print("np.c_[xx.ravel(), yy.ravel()] : ",np.c_[xx.ravel(), yy.ravel()])

	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
	print("Z : ",Z)
	Z = np.array(Z)[:,0]
	#print("Z : ",Z)
	Z_norm = [(float(i)-min(Z))/(max(Z)-min(Z)) for i in Z]
	#if you want the pure output of the machine learning algorithm uncomment the following line
	Z = Z_norm 
	Z = np.array(Z)
	print("Z : ",Z)
	Z = Z.reshape(xx.shape)
	print("Z : ",Z)
    	cs = plt.contourf(xx, yy, Z)
	plt.colorbar()
	plt.title("Visualisation of decision boundary normalised")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig("visualisation.png")

	#plt.figure()
	#plt.pcolor(xx,yy,Z)
	#plt.colorbar()
	#plt.savefig("visualisation2.png")
	
	print(p_KS_stat)
        p_KS=-p_KS_stat[1]
        return p_KS

def p_value_scoring_object_AD(clf, X, y): 
        """ 
        p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
        """

        #Finding out the prediction probabilities
        prob_pred=clf.predict_proba(X)[:,1]
        #print(prob_pred)

	#This can be deleted if not using Keras
        #For Keras turn cathegorical y back to normal y
        if y.ndim==2:
                if y.shape[0]!=1 and y.shape[1]!=1:
                        #Then we have a cathegorical vector
                        y = y[:,1]

        #making sure the inputs are row vectors
        y         = np.reshape(y,(1,y.shape[0]))
        prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

        #Separate prob into particle and antiparticle samples
        prob_0    = prob_pred[np.logical_or.reduce([y==0])]
        prob_1    = prob_pred[np.logical_or.reduce([y==1])]
        #if __debug__:
                #print("Plot")
        p_AD_stat=stats.anderson_ksamp([prob_0,prob_1])
        print(p_AD_stat)
        p_AD=-p_AD_stat[2]
        return p_AD



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
	
	print("KS p value : ",p_value_scoring_object(clf,X[50:,:],y[50:]))
	print("AD p value : ",p_value_scoring_object_AD(clf,X[50:,:],y[50:]))


