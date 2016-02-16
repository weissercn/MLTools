#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     classifier_eval.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a class structure to determine the p value for different
#           scikit-learn algorithms. 
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 


from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
import tensorflow as tf

from two_sample_tests import cramer_von_mises_2sample as cr

from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import cross_validation
from sknn.mlp import Classifier, Layer
from scipy import stats


class classifier(object):
    """
    A class that prepares the data, trains a classifier on it and returns the success of it. 
    Its methods are __init__, get_results, get_results_without_cross_validation, reset, scale_data, 
    train_from_scratch, train, get_percentage_wrong, get_score_without_cross_validation, get_score, 
    get_pvalue_perm, ks, set_percentage_used_for_primary, set_no_permutations, 
    get_percentage_used_for_primary , get_no_permutations , get_clf, get_score_list and print_line
    """
    print(__doc__)
    #clf = tree.DecisionTreeClassifier()
    
    def __init__(self,data_unscaled,percentage_used_for_validation,no_permutations=0):
        
        #Reset everything, because I don't really understand destructors in python. Just to be sure.
        self.reset()
        
        #Make the inputs class variables
        self.percentage_used_for_validation = percentage_used_for_validation
        self.no_permutations = no_permutations
        self.data_unscaled = data_unscaled
        
        # Scaling the data in case the typical distance of the origin of the Dalitz plot is not of order 1
        self.scale_data()
        
        #Defining primary and validation data
        self.data_primary=self.data[:math.floor(self.data.shape[0]*(1-percentage_used_for_validation/100)),:]
        self.data_validation=self.data[math.floor(self.data.shape[0]*(1-percentage_used_for_validation/100)):,:]

        #Selecting features (X) and labels (y)
        self.X_pri = self.data_primary[:,:-1]
        self.no_primary=self.X_pri.shape[0]
        self.y_pri = self.data_primary[:,2:].reshape((self.no_primary,))
	self.y_pri_tf = np.zeros(( self.no_primary,2))

	for i in range(self.no_primary):
    		if self.y_pri[i]==1:
        		self.y_pri_tf[i,1]=1
    		else:
        		self.y_pri_tf[i,0]=1	
        
        self.X_val = self.data_validation[:,:-1]
        self.no_validation=self.X_val.shape[0]
        self.y_val = self.data_validation[:,2:].reshape((self.no_validation,))
	self.y_val_tf = np.zeros(( self.no_validation,2))

	for i in range(self.no_validation):
    		if self.y_val[i]==1:
        		self.y_val_tf[i,1]=1
    		else:
        		self.y_val_tf[i,0]=1


        self.print_line()
        
        
        
    def get_results_without_cross_validation(self):
        # This is called to train the algorithm and judge its success. 
        # Training is not needed as it is included in the score function
        self.train_from_scratch()
        #self.get_score_without_cross_validation()
        #self.get_percentage_wrong()
        #self.get_pvalue_perm_without_cross_validation(self.no_permutations)
        self.ks()
        
    def get_results(self):
        # Because this method uses crossvalidation no prior training is required 
        # as this is done in the crossvalidation function.
        self.clf = self.clf_blueprint
	self.get_score()
        self.get_percentage_wrong()
        self.get_pvalue_perm(self.no_permutations)
        self.ks()
    
    def reset(self):
        self.score_list=[]
        self.percentage_used_for_validation=-1
        self.no_permutations=-1
        self.data_primary=-1
        self.data_validation=-1
        self.no_primary=-1
        self.no_validation=-1
        self.X_pri=-1
        self.y_pri=-1
	self.y_pri_tf=-1
        self.X_val=-1
        self.y_val=-1
        self.y_val_tf=-1
	self.percentage_wrong_primary=-1
        self.percentage_wrong_validation=-1
        self.score_primary=-1
        self.score_validation=-1
        self.score=-1
        
    def scale_data(self):
        #This function scales the data 
        self.data=self.data_unscaled
        #get the unscaled training features as in __init__
        X_primary_unscaled = self.data[:math.floor(self.data_unscaled.shape[0]*(1-self.percentage_used_for_validation/100)),:][:,:-1]
        scaler = preprocessing.StandardScaler().fit(X_primary_unscaled)
        self.data[:,:-1]=scaler.fit_transform(self.data[:,:-1])
        return scaler
    
       
    def get_percentage_wrong(self):
        """Need to run train_from_scratch or train before this"""
        #Very quick check how well the algorithm did. 
        
        #prediction list: 0 is correct, 1 if wrong prediction
        pred_primary = abs(self.clf.predict(self.X_pri)-self.y_pri)
        pred_validation = abs(self.clf.predict(self.X_val)-self.y_val)

        # Find the number of wrong predictions and divide by the number of samples. 
        # The result is the percentage of wrong classifications.
        self.percentage_wrong_primary= np.sum(pred_primary)/self.no_primary*100
        self.percentage_wrong_validation= np.sum(pred_validation)/self.no_validation*100

        print("Percentage of wrong predictions Training: %.4f%%" % self.percentage_wrong_primary)
        print("Percentage of wrong predictions Validation: %.4f%%" % self.percentage_wrong_validation)
        
        return (self.percentage_wrong_primary,self.percentage_wrong_validation)
    
    def get_score_without_cross_validation(self):
        """Should run train before get_score"""
        # Using the score function of each algorithm
        self.score_primary = self.clf.score(self.X_pri, self.y_pri)
        self.score_validation = self.clf.score(self.X_val, self.y_val)
        print("Training score: %.4f" % self.score_primary)
        print("Validation score: %.4f (used as score)" % self.score_validation)
        self.score = self.score_validation
        return self.score
    
    def get_score(self):
        # Firstly the accuracy on the primary sample is found, the algorithm is then TRAINED on the WHOLE 
        # primary sample and then predictions about the validation sample made
        """Should run train before get_score"""
        # Using the score function of each algorithm
        # Unlike get_score_without_cross_validation this function divides the training sample into 
        # a training and a testing sample. It still uses the validation sample as well.
        self.scores_primary = cross_validation.cross_val_score(self.clf, self.X_pri, self.y_pri, cv=4)
        print("Accuracy: %0.2f (+/- %0.2f)" % (self.scores_primary.mean(), self.scores_primary.std() * 2))
        self.score_primary = self.scores_primary.mean()
        self.train_from_scratch()
        self.score_validation = self.clf.score(self.X_val, self.y_val)
        print("Testing score: %.4f (used as score)" % self.score_primary)
        print("Validation score: %.4f" % self.score_validation)
        self.score= self.score_primary
        return self.score
    
    def get_pvalue_perm_without_cross_validation(self,no_permutations):
        """Need to run get_score_without_cross_validation before get_pvalue_perm_without_cross_validation"""
        if(no_permutations==0):
            return -1
        
        print("score_list")
        self.score_list=[]
        print(self.score_list)
        self.print_line()
        print("Performing %f permutations" %self.no_permutations)
        
        
        for i in range(0,no_permutations):
            y_pri_shuffled=np.random.permutation(self.y_pri)
            y_val_shuffled=np.random.permutation(self.y_val)
            clf_shuffled = self.clf_blueprint
            clf_shuffled = clf_shuffled.fit(self.X_pri, y_pri_shuffled)
            self.score_list.append(clf_shuffled.score(self.X_val, y_val_shuffled))

        print("Self score: %.4f" % self.score)
        print(self.score_list)
        self.pvalue=sum(i > self.score for i in self.score_list)/self.no_permutations
        print(self.pvalue)
        
        self.print_line()
        return self.pvalue
    
    def get_pvalue_perm(self,no_permutations):
        """Need to run get_score before get_pvalue_perm"""
        if(no_permutations==0):
            return -1
        
        print("score_list")
        self.score_list=[]
        print(self.score_list)
        self.print_line()
        print("Performing %f permutations" %self.no_permutations)
        
        
        for i in range(0,no_permutations):
            y_pri_shuffled=np.random.permutation(self.y_pri)
            #y_val_shuffled=np.random.permutation(self.y_val)
            clf_shuffled = self.clf_blueprint
            #clf_shuffled = clf_shuffled.fit(self.X_pri, y_pri_shuffled)
            self.score_list.append(cross_validation.cross_val_score(clf_shuffled, self.X_pri, y_pri_shuffled, cv=4).mean())

        print("Self score: %.4f" % self.score)
        print(self.score_list)
        self.pvalue=sum(i > self.score for i in self.score_list)/self.no_permutations
        print(self.pvalue)
        
        self.print_line()
        return self.pvalue
    
    def ks(self):
        # It shouldnt matter if k folding was performed or not. the algorythm should be trained on the whole primary sample.
        #self.train_from_scratch()
        #prediction list: 0 is correct, 1 if wrong prediction
	pred_validation = abs(self.clf.predict(self.X_val)-self.y_val)
        
        # Use predict_proba to find the probability that a point came from file 1.  
        probability_from_file_1 = self.clf.predict_proba(self.X_val)[:,1]   
        pred_validation_transposed= np.reshape(pred_validation,(self.no_validation,1))
        probability_from_file_1_transposed= np.reshape(probability_from_file_1,(self.no_validation,1))

        self.data_validation_with_pred = np.c_[self.data_validation, pred_validation_transposed,probability_from_file_1_transposed]
        self.data_validation_file_0 = self.data_validation_with_pred[np.logical_or.reduce([self.data_validation_with_pred[:,2] ==0])]
        self.data_validation_file_1 = self.data_validation_with_pred[np.logical_or.reduce([self.data_validation_with_pred[:,2] ==1])]
        
        print("self.data_validation_file_0")
        print(self.data_validation_file_0)
        
        pred_validation_file_0=self.data_validation_file_0[:,3]
        pred_validation_file_1=self.data_validation_file_1[:,3]
        
        bins_probability=np.histogram(np.hstack((self.data_validation_file_0[:,4],self.data_validation_file_1[:,4])), bins=500)[1]
        
        # Making a histogram of the probability predictions of the algorithm. 
        fig_pred_0= plt.figure()
	ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
	n0, bins0, patches0 = ax1_pred_0.hist(self.data_validation_file_0[:,4], bins=bins_probability, normed=True, facecolor='red', alpha=0.5, label="file_0")
        ax1_pred_0.set_xlabel('Probability')
        ax1_pred_0.set_ylabel('Normalised Frequency')
        ax1_pred_0.set_title('Probability Predictions Particle Sample BDT')
        fig_pred_0.savefig('Machine_learning_predictions_particle_BDT.pdf', format='pdf', dpi=300)
        #ax1_pred_0.show()
        
	fig_pred_1= plt.figure()
	ax1_pred_1= fig_pred_1.add_subplot(1, 1, 1)
	n1, bins1, patches1 = ax1_pred_1.hist(self.data_validation_file_1[:,4], bins=bins_probability, normed=True, facecolor='blue', alpha=0.5, label="file_1")
        #plt.axis([0, 1, 0, 0.03])
        ax1_pred_1.set_xlabel('Probability')
        ax1_pred_1.set_ylabel('Normalised Frequency')
        ax1_pred_1.set_title('Probability Predictions CPV2 Sample BDT')
        fig_pred_1.savefig('Machine_learning_predictions_CPV2_BDT.pdf', format='pdf', dpi=300)
        #ax1_pred_1.show()
        
        fig_pred_comp= plt.figure()
	ax1_pred_comp= fig_pred_comp.add_subplot(1, 1, 1)
        n0, bins0, patches0 = ax1_pred_comp.hist(self.data_validation_file_0[:,4], bins=bins_probability, normed=True, facecolor='red', alpha=0.5, label="Particle")
        n1, bins1, patches1 = ax1_pred_comp.hist(self.data_validation_file_1[:,4], bins=bins_probability, normed=True, facecolor='blue', alpha=0.5, label="Antiparticle")
        #plt.axis([0.46, 0.53,0,600])
        ax1_pred_comp.legend(loc='upper right')
        ax1_pred_comp.set_xlabel('Probability')
        ax1_pred_comp.set_ylabel('Normalised Frequency')
        ax1_pred_comp.set_title('Probability Predictions Particle and CPV2 Samples BDT')
        fig_pred_comp.savefig('Machine_learning_predictions_particle_CPV2_BDT.pdf', format='pdf', dpi=3000)
        #ax1_pred_comp.show()
        
        
        #Subtract histograms. This is assuming equal bin width
        fig_subt= plt.figure()
	ax1_subt= fig_subt.add_subplot(1,1,1)	

	ax1_subt.scatter(bins_probability[:-1]+(bins_probability[1]-bins_probability[0])/2,n1-n0,facecolors='blue', edgecolors='blue')
        ax1_subt.set_xlabel('Probability')
        ax1_subt.set_ylabel('Normalised Frequency (sample 1 - sample 2)')
        ax1_subt.set_title('Differtial Probability Predictions Particle and CPV2 Samples BDT')
        fig_subt.savefig('Machine_learning_predictions_CPV2_minus_particle_BDT.pdf', format='pdf', dpi=3000)
        #ax1_subt.show()
        
        #val[numpy.logical_or.reduce([val[:,1] == 1])]
        colorMap = plt.get_cmap('Spectral')
        fig_dalitz_color= plt.figure()
	ax1_dalitz_color= fig_dalitz_color.add_subplot(1,1,1)
        ax1_dalitz_color.scatter( self.data_validation_file_0[:,0],self.data_validation_file_0[:,1],10,self.data_validation_file_0[:,3],cmap=colorMap)
        fig_dalitz_color.savefig('validation_file_0.pdf', format='pdf', dpi=300)
        
        print("pred_validation_file_0")
        print(pred_validation_file_0)
        print("pred_validation_file_1")
        print(pred_validation_file_1)
        result_KS=stats.ks_2samp(self.data_validation_file_0[:,4], self.data_validation_file_1[:,4])
        print("KS result (KS statistic, p value)")
	print(result_KS)
        
	result_CvM= cr.cvm_2samp(self.data_validation_file_0[:,4], self.data_validation_file_1[:,4])
	print("Cramer von Mises result (U value, T value)")
	print(result_CvM)


	return result_KS[1]
        
        
    def set_percentage_used_for_primary(self,percentage_used_for_primary):
        self.percentage_used_for_primary = percentage_used_for_primary
        
    def set_no_permutations(self,no_permutations):
        self.no_permutations = no_permutations
        
    def get_percentage_used_for_primary(self):
        return self.percentage_used_for_primary
    
    def get_no_permutations(self):
        return self.no_permutations
    
    def get_clf(self):
        return self.clf
    
    def get_score_list(self):
        return self.score_list
    
    def print_line(self):
        print("---------------------------------------------------------------------------------------------------------------------")

####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
        

class sklearn_classifier(classifier):
	def train_from_scratch(self):
        	self.clf = self.clf_blueprint
        	print(self.clf)
		self.print_line()
		self.clf = self.clf.fit(self.X_pri, self.y_pri)
        
    	def train(self):
        	self.clf = self.clf.fit(self.X_pri, self.y_pri)
	
	type_of_classifier="sklearn"

class tf_classifier(classifier):
	type_of_classifier="tensorflow"

####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################




class softmax_regression_tf(tf_classifier):
	def train_from_scratch(self):
		sess = tf.InteractiveSession()
		x = tf.placeholder("float", shape=[None, 2])
		y_ = tf.placeholder("float", shape=[None, 2])
		W = tf.Variable(tf.zeros([2,2]))
		b = tf.Variable(tf.zeros([2]))

		#print("test2")

		sess.run(tf.initialize_all_variables())
		y = tf.nn.softmax(tf.matmul(x,W) + b)
		cross_entropy = -tf.reduce_sum(y_*tf.log(y))
		train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

		#print("test3")
		#print(X_pri)

		for i in range(self.no_primary):
		    #print(X_pri[i,:].reshape(1,2).shape)
		    #print(y_pri_tf[i,:].reshape(1,2).shape)
		    train_step.run(feed_dict={x: self.X_pri[i,:].reshape(1,2), y_: self.y_pri_tf[i,:].reshape(1,2)})

		#print("test4")    
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#print("test5")
		print("accuracy on validation sample")
		print(accuracy.eval(feed_dict={x: self.X_val[:,:].reshape(self.no_validation,2), y_: self.y_val_tf[:,:].reshape(self.no_validation,2)}))
		self.print_line()


class dt_sklearn(sklearn_classifier):
    clf_blueprint = tree.DecisionTreeClassifier()
    
class ada_sklearn(sklearn_classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0, no_estimators=1000):
        self.no_estimators = no_estimators
        print("no_estimators: %.4f" % self.no_estimators)
        self.clf_blueprint = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=no_estimators)
        super( obdt, self ).__init__(data,percentage_used_for_validation,no_permutations)
    def set_no_estimators(self,no_estimators):
        self.no_estimators=no_estimators
    def get_no_estimators(self):
        return self.no_estimators
    
class svm_sklearn(sklearn_classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0, a_cache_size=1000):
        self.cache_size = a_cache_size
        print("cache_size: %.4f" % self.cache_size)
        self.clf_blueprint = svm.SVC(probability=True,cache_size=a_cache_size)
        super( osvm, self ).__init__(data,percentage_used_for_validation,no_permutations)
    def set_cache_size(self,a_cache_size):
        self.cache_size=a_cache_size
    def get_cache_size(self):
        return self.cache_size
    
class nn_sklearn(sklearn_classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0):
        self.clf_blueprint = Classifier(
            layers=[
                Layer("Rectifier", units=10),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=1)
        print(self.clf_blueprint)
        super( onn, self ).__init__(data,percentage_used_for_validation,no_permutations)







