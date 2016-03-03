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

# __debug__ = 1 if python is  not started with -o 

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

from keras.models import Sequential
from keras.layers.core import Dense, Activation


class classifier(object):
    """
    A class that prepares the data, trains a classifier on it and returns the success of it. 
    Its methods are __init__, get_results, get_results_without_cross_validation, reset, scale_data, 
    train_from_scratch, train, get_percentage_wrong, get_score_without_cross_validation, get_score, 
    get_pvalue_perm, ks, set_percentage_used_for_primary, set_no_permutations, 
    get_percentage_used_for_primary , get_no_permutations , get_clf, get_score_list and print_line
    """
    if __debug__:print(__doc__)
    #clf = tree.DecisionTreeClassifier()
    
    def __init__(self,data_unscaled,percentage_used_for_validation,no_permutations=0,name="unnamed",sample1_name="sample1",sample2_name="sample2"):
        #Reset everything, because I don't really understand destructors in python. Just to be sure.
        self.reset()
        
	self.name=name
	self.sample1_name=sample1_name
	self.sample2_name=sample2_name

        #Make the inputs class variables
        self.percentage_used_for_validation = percentage_used_for_validation
        self.no_permutations = no_permutations
        self.data_unscaled = data_unscaled
        
        # Scaling the data in case the typical distance of the origin of the Dalitz plot is not of order 1
	self.data=self.data_unscaled
        #self.scale_data()
        
        #Defining primary and validation data
        self.data_primary=self.data[:math.floor(self.data.shape[0]*(1-percentage_used_for_validation/100)),:]
        self.data_validation=self.data[math.floor(self.data.shape[0]*(1-percentage_used_for_validation/100)):,:]

	if __debug__:
		print("self.data_primary")
		print(self.data_primary)
		print("self.data_validation")
		print(self.data_validation)

        #Selecting features (X) and labels (y)
        self.X_pri = self.data_primary[:,:-1]
        self.no_primary=self.X_pri.shape[0]
        self.y_pri = self.data_primary[:,2:].reshape((self.no_primary,))
	self.y_pri_tf = np.zeros(( self.no_primary,2))

	#Primary sample Dalitz Plot
	plt.rc('text', usetex=True)
	dalitz_pri=plt.figure()
	ax_dal=dalitz_pri.add_subplot(1,1,1)
	ax_dal.scatter(self.X_pri[:,0],self.X_pri[:,1],s=0.2)
	ax_dal.set_xlabel(r'$m_{AB}^2$')
	ax_dal.set_ylabel(r'$m_{AC}^2$')
	ax_dal.set_title("Dalitz plot")
	dalitz_pri.savefig("Dalitz_plot")

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
        
        
        
    def get_results(self):
        # This is called to train the algorithm and judge its success. 
        # Training is not needed as it is included in the score function
        self.train_from_scratch()
        #self.get_score()
        #self.get_percentage_wrong()
        #self.get_pvalue_perm(self.no_permutations)
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
        #get the unscaled training features as in __init__
        X_primary_unscaled = self.data[:math.floor(self.data_unscaled.shape[0]*(1-self.percentage_used_for_validation/100)),:][:,:-1]
        scaler = preprocessing.StandardScaler().fit(X_primary_unscaled)
        self.data[:,:-1]=scaler.fit_transform(self.data[:,:-1])
        return scaler
    
       
    def get_percentage_wrong(self):
        """Need to run train_from_scratch or train before this"""
        #Very quick check how well the algorithm did. 
        
        #prediction list: 0 is correct, 1 if wrong prediction
        pred_primary = abs(self.predict(self.X_pri)-self.y_pri)
        pred_validation = abs(self.predict(self.X_val)-self.y_val)

        # Find the number of wrong predictions and divide by the number of samples. 
        # The result is the percentage of wrong classifications.
        self.percentage_wrong_primary= np.sum(pred_primary)/self.no_primary*100
        self.percentage_wrong_validation= np.sum(pred_validation)/self.no_validation*100

	if __debug__:
        	print("Percentage of wrong predictions Training: %.4f%%" % self.percentage_wrong_primary) 
        	print("Percentage of wrong predictions Validation: %.4f%%" % self.percentage_wrong_validation)
        
        return (self.percentage_wrong_primary,self.percentage_wrong_validation)
    
    def get_pvalue_perm_percentage_wrong(self,no_permutations):
        """Need to run get_score before get_pvalue_perm_percentage_wrong"""
        if(no_permutations==0):
            print("no_permutations was 0. Terminating")
            return -1

	if __debug__:
		self.print_line()	
		print("get_pvalue_perm_percentage_wrong is not completely implemented.")
		print(" Need to use get_percentage_wrong functon and be compatible with tf and sklean")
        	print("Calculating p value from permutation")
        	print("score_list")
        self.score_list=[]

	if __debug__:
        	print(self.score_list)
        	print("Performing %f permutations" %no_permutations)
    
    
        for i in range(0,no_permutations):
            y_pri_shuffled=np.random.permutation(self.y_pri)
            y_val_shuffled=np.random.permutation(self.y_val)
            clf_shuffled = self.clf_blueprint
            clf_shuffled = clf_shuffled.fit(self.X_pri, y_pri_shuffled)
            self.score_list.append(clf_shuffled.score(self.X_val, y_val_shuffled))

	self.pvalue=sum(i > self.score for i in self.score_list)/self.no_permutations

	if __debug__:
        	print("Self score: %.4f" % self.score)
        	print(self.score_list)
        	print(self.pvalue)
        	self.print_line()
        return self.pvalue
    
    def ks(self):
        # It shouldnt matter if k folding was performed or not. the algorythm should be trained on the whole primary sample.
        #self.train_from_scratch()
        #prediction list: 0 is correct, 1 if wrong prediction
	predicted_y_val = self.predict(self.X_val)
	if(self.specific_type_of_classifier.startswith("nn")):
		predicted_y_val=np.reshape(predicted_y_val,(1,self.no_validation) )
	pred_validation = abs(predicted_y_val-self.y_val)

	if __debug__:	
		print("self.predict(self.X_val)")
		print(self.predict(self.X_val))
		print("len(pred_validation)")
		print(len(pred_validation))
		print(pred_validation)
		print("self.no_validation")
		print(self.no_validation)
 
        # Use predict_proba to find the probability that a point came from file 1.  
        probability_from_file_1 = self.predict_proba(self.X_val)[:,1]   
        pred_validation_transposed= np.reshape(pred_validation,(self.no_validation,1))
        probability_from_file_1_transposed= np.reshape(probability_from_file_1,(self.no_validation,1))

	if __debug__:
		print("pred_validation_transposed")
		print(pred_validation_transposed)
		print("probability_from_file_1")
		print(probability_from_file_1)

	# colums of self.data_validation: x, y, label, prediction, prediction probability 
        self.data_validation_with_pred = np.c_[self.data_validation, pred_validation_transposed,probability_from_file_1_transposed]
        self.data_validation_file_0 = self.data_validation_with_pred[np.logical_or.reduce([self.data_validation_with_pred[:,2] ==0])]
        self.data_validation_file_1 = self.data_validation_with_pred[np.logical_or.reduce([self.data_validation_with_pred[:,2] ==1])]

	if __debug__:
		print("self.data_validation")
		print(self.data_validation)
        	print("self.data_validation_with_pred")
        	print(self.data_validation_with_pred)

	        print("self.data_validation_file_0")
        	print(self.data_validation_file_0)

        	print("self.data_validation_file_1")
        	print(self.data_validation_file_1)

        
        pred_validation_file_0=self.data_validation_file_0[:,3]
        pred_validation_file_1=self.data_validation_file_1[:,3]

	if __debug__:
		print("pred_validation_file_0")
		print(pred_validation_file_0)

        	print("pred_validation_file_1")
        	print(pred_validation_file_1)

	if __debug__:
	        bins_probability=np.histogram(np.hstack((self.data_validation_file_0[:,4],self.data_validation_file_1[:,4])), bins=50)[1]
        
        	# Making a histogram of the probability predictions of the algorithm. 
        	fig_pred_0= plt.figure()
		ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
		n0, bins0, patches0 = ax1_pred_0.hist(self.data_validation_file_0[:,4], bins=bins_probability, facecolor='red', alpha=0.5, label="file_0")
        	ax1_pred_0.set_xlabel('Probability')
        	ax1_pred_0.set_ylabel('Normalised Frequency')
        	ax1_pred_0.set_title('Probability Predictions '+self.name+' '+self.sample1_name+' '+self.specific_type_of_classifier)
        	fig_pred_0.savefig('graphs/Machine_learning_predictions_'+self.name+'_'+self.sample1_name+'_'+self.specific_type_of_classifier+'.pdf', format='pdf', dpi=300)
        	#ax1_pred_0.show()
        
		fig_pred_1= plt.figure()
		ax1_pred_1= fig_pred_1.add_subplot(1, 1, 1)
		n1, bins1, patches1 = ax1_pred_1.hist(self.data_validation_file_1[:,4], bins=bins_probability, facecolor='blue', alpha=0.5, label="file_1")
        	#plt.axis([0, 1, 0, 0.03])
        	ax1_pred_1.set_xlabel('Probability')
        	ax1_pred_1.set_ylabel('Normalised Frequency')
        	ax1_pred_1.set_title('Probability Predictions '+self.name+' '+self.sample2_name+' '+self.specific_type_of_classifier)
        	fig_pred_1.savefig('graphs/Machine_learning_predictions_'+self.name+'_'+self.sample2_name+'_'+self.specific_type_of_classifier+'.pdf', format='pdf', dpi=300)
        	#ax1_pred_1.show()
        
        	fig_pred_comp= plt.figure()
		ax1_pred_comp= fig_pred_comp.add_subplot(1, 1, 1)
        	n0, bins0, patches0 = ax1_pred_comp.hist(self.data_validation_file_0[:,4], bins=bins_probability, facecolor='red', alpha=0.5, label="Particle")
        	n1, bins1, patches1 = ax1_pred_comp.hist(self.data_validation_file_1[:,4], bins=bins_probability, facecolor='blue', alpha=0.5, label="Antiparticle")
        	#plt.axis([0.46, 0.53,0,600])
        	ax1_pred_comp.legend(loc='upper right')
        	ax1_pred_comp.set_xlabel('Probability')
        	ax1_pred_comp.set_ylabel('Normalised Frequency')
        	ax1_pred_comp.set_title('Probability Predictions '+self.name+' '+self.sample1_name+' and '+self.sample2_name+' '+self.specific_type_of_classifier)
        	fig_pred_comp.savefig('graphs/Machine_learning_predictions_'+self.name+'_'+self.sample1_name+'_'+self.sample2_name+'_'+self.specific_type_of_classifier+'.pdf', format='pdf', dpi=3000)
        	#ax1_pred_comp.show()
        
        
        	#Subtract histograms. This is assuming equal bin width
        	fig_subt= plt.figure()
		ax1_subt= fig_subt.add_subplot(1,1,1)	

		ax1_subt.scatter(bins_probability[:-1]+(bins_probability[1]-bins_probability[0])/2,n1-n0,facecolors='blue', edgecolors='blue')
        	ax1_subt.set_xlabel('Probability')
        	ax1_subt.set_ylabel('Normalised Frequency'+self.name+' ( '+self.sample1_name+' - '+ self.sample2_name+' )' )
        	ax1_subt.set_title('Differtial Probability Predictions '+self.name+' '+self.sample1_name+' and '+self.sample2_name+' '+self.specific_type_of_classifier)
        	fig_subt.savefig('graphs/Machine_learning_predictions_'+self.name+'_'+self.sample1_name+'_minus_'+self.sample2_name+'_'+self.specific_type_of_classifier+'.pdf', format='pdf', dpi=3000)
        	#ax1_subt.show()
        
        	#val[numpy.logical_or.reduce([val[:,1] == 1])]
        	colorMap = plt.get_cmap('Spectral')
        	fig_dalitz_color= plt.figure()
		ax1_dalitz_color= fig_dalitz_color.add_subplot(1,1,1)
        	ax1_dalitz_color.scatter( self.data_validation_file_0[:,0],self.data_validation_file_0[:,1],10,self.data_validation_file_0[:,3],cmap=colorMap)
        	fig_dalitz_color.savefig('graphs/validation_file_0_'+self.name+'_'+self.sample1_name+'_and_'+self.sample2_name+'_'+self.specific_type_of_classifier+'.pdf', format='pdf', dpi=300)
        
	result_KS=stats.ks_2samp(self.data_validation_file_0[:,4], self.data_validation_file_1[:,4])
	result_CvM= cr.cvm_2samp(self.data_validation_file_0[:,4], self.data_validation_file_1[:,4])
	#result_perm = self.get_pvalue_perm_score(self.no_permutations)

	if __debug__:
		print("self.data_validation_file_0[:,4]")
		print(self.data_validation_file_0[:,4])
		print("self.data_validation_file_1[:,4]")
		print(self.data_validation_file_1[:,4])
 
		print("KS result (KS statistic, p value)")
		print(result_KS)
		print("Cramer von Mises result (U value, T value)") 
		print(result_CvM)
		print("P value from get_pvalue_perm_score")
	
	with open("test_statistic_distributions/test_statistics_"+self.name+"_"+self.sample1_name+"_"+self.sample2_name+"_"+self.type_of_classifier+"_"+self.specific_type_of_classifier, "a") as test_statistics_file:
		test_statistics_file.write("{0} \t{1} \t{2} \t{3} \n".format(result_CvM[0],result_CvM[1],result_KS[0], result_KS[1]))

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
	
	def predict(self, X_val):
		return self.clf.predict(X_val)

	def predict_proba(self, X_val):
		return self.clf.predict_proba(X_val)

	       
	def get_score(self):
		"""Should run train before get_score"""
		# Using the score function of each algorithm
		self.score_primary = self.clf.score(self.X_pri, self.y_pri)
		self.score_validation = self.clf.score(self.X_val, self.y_val)
		print("Training score: %.4f" % self.score_primary)
		print("Validation score: %.4f (used as score)" % self.score_validation)
		self.score = self.score_validation
		return self.score

	def get_pvalue_perm_score(self,no_permutations):
		"""Need to run get_score before get_pvalue_perm_score"""
		if(no_permutations==0):
			print("no_permutations was 0. Terminating")
			return -1
       
		self.print_line() 
		print("Calculating p value from permutation of score")
		print("score_list")
		self.score_list=[]
		print(self.score_list)
		print("Performing %f permutations" %no_permutations)
        
        
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

	type_of_classifier="sklearn"

class tf_classifier(classifier):

	def train_from_scratch(self):
		print("Need to implement the method TRAIN_FROM_SCRATCH")

	def predict(self, X_val):
		print("Need to implement the method PREDICT")

	def predict_proba(self, X_val):
		print("Need to implement the method PREDICT_PROBA")


	type_of_classifier="tensorflow"

class keras_classifier(classifier):

        def train_from_scratch(self):
		#Setting up a sequential model rather than a Graph
		self.model = Sequential()

		#Adding layers
		self.model.add(Dense(output_dim=1000, input_dim=2, init="glorot_uniform",activation='relu'))

                self.model.add(Dense(output_dim=1000, input_dim=1000, init="glorot_uniform",activation='tanh'))
                self.model.add(Dense(output_dim=1000, input_dim=1000, init="glorot_uniform",activation='tanh'))
                self.model.add(Dense(output_dim=1000, input_dim=1000, init="glorot_uniform",activation='tanh'))
                self.model.add(Dense(output_dim=1000, input_dim=1000, init="glorot_uniform",activation='tanh'))

		self.model.add(Dense(output_dim=2, init="glorot_uniform",activation='sigmoid'))

		#Compiling (might take longer)
		self.model.compile(loss='categorical_crossentropy', optimizer='sgd')
		
		#specifying stochastic gradient descent parameters
                #from keras.optimizers import SGD 
                #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		#self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

		from keras.utils import np_utils, generic_utils
		self.y_pri_cathegorical=(np_utils.to_categorical(self.y_pri))

		x_delete=np.array([[1, 2], [3, 4],[5, 6]])
		y_delete=np.array([[1],[3],[5]])

		#training
		#model.train_on_batch(x_delete, y_delete)
		self.model.train_on_batch(self.X_pri, self.y_pri_cathegorical)

		#model.fit(self.X_pri, self.y_pri, nb_epoch=10, batch_size=100)
		

        def predict(self, X_val):
		return self.model.predict_classes(X_val)


        def predict_proba(self, X_val):
		return self.model.predict_proba(X_val)

	def get_pvalue_perm_score(self,no_permutations):
		if __debug__: print("get_pvalue_perm_score not implemented, yet")

        type_of_classifier="keras"
	specific_type_of_classifier="dense_activation_4_hidden_1000neurons_tanh"

class twodim_miranda(classifier):
	def __init__(self,data,percentage_used_for_validation,no_permutations=0, no_binsx=3, no_binsy=3,name="unnamed",sample1_name="sample1",sample2_name="sample2"):
		self.no_binsx=no_binsx
		self.no_binsy=no_binsy
		super( twodim_miranda, self ).__init__(data,percentage_used_for_validation,no_permutations,name,sample1_name,sample2_name)
	def get_results(self):
		import numpy.matlib
		self.bins_sample0=np.matlib.zeros((self.no_binsy,self.no_binsx))
		self.bins_sample1=np.matlib.zeros((self.no_binsy,self.no_binsx))
		Xx_values=self.data[:,0]
		Xy_values=self.data[:,1]
		labels=self.data[:,2]
		no_binsx=self.no_binsx
		no_binsy=self.no_binsy

		Xx_max= np.amax(Xx_values)
		Xx_min= np.amin(Xx_values)
		Xx_width= (Xx_max-Xx_min)/no_binsx
		Xy_max= np.amax(Xy_values)
		Xy_min= np.amin(Xy_values)
		Xy_width= (Xy_max-Xy_min)/no_binsy

		for i in range(Xx_values.shape[0]):
			x_bin=int(np.floor((Xx_values[i]-Xx_min)/Xx_width))	
			y_bin=int(np.floor((Xy_values[i]-Xy_min)/Xy_width))
			if(x_bin==no_binsx):
				x_bin -=1
			if(y_bin==no_binsy):
				y_bin -=1
			if(labels[i]==0):
				self.bins_sample0[y_bin,x_bin] +=1
			else:
				self.bins_sample1[y_bin,x_bin] +=1

		print(self.bins_sample0)
		print(self.bins_sample0[1,1])
		print(np.sum(self.bins_sample0))

                print(self.bins_sample1)
                print(self.bins_sample1[1,1])
                print(np.sum(self.bins_sample1))
		
		#element wise subtraction and division
		Scp2 =  np.divide(np.square(np.subtract(self.bins_sample1,self.bins_sample0)),np.add(self.bins_sample1,self.bins_sample0))
		print(Scp2)
		Chi2 = np.nansum(Scp2)
		print("Chi2")
		print(Chi2)

		print("Chi2/dof")
		print(Chi2/(no_binsx*no_binsy-1))

		pvalue= 1 - stats.chi2.cdf(Chi2,(no_binsx*no_binsy-1))
		print("pvalue")
		print(pvalue)

		with open("test_statistic_distributions/test_statistics_"+self.name+"_"+self.sample1_name+"_"+self.sample2_name+"_miranda_"+str(self.no_binsx)+"_"+str(self.no_binsy), "a") as test_statistics_file:
			test_statistics_file.write("{0} \t{1} \t{2} \t{3} \n".format(0,0,0,pvalue))


class twodim_energy_test(classifier):
        def __init__(self,data,percentage_used_for_validation,no_permutations, sigma,features_0,features_1,name="unnamed",sample1_name="sample1",sample2_name="sample2"):
                self.sigma=sigma
		self.features_0=features_0
		self.features_1=features_1
                super( twodim_energy_test, self ).__init__(data,percentage_used_for_validation,no_permutations,name,sample1_name,sample2_name)
        
	def weighting_function(self,dx):
		return np.exp(-np.square(dx)/(2*np.square(self.sigma)))

	def distance_squared(self,features_0,features_1,i,j):
		dx2=0
		assert features_0.shape[1]==features_1.shape[1]
		for d in range(features_0.shape[1]):
			dx2 += np.square(features_1[j,d]-features_0[i,d])
		dx=np.sqrt(dx2)
		return self.weighting_function(dx2)

	def get_results(self):
		features_0=self.features_0
		features_1=self.features_1	
	
		no_0=features_0.shape[0]
		no_1=features_1.shape[0]
		T_1st_contrib=0

		for i in range(no_0):
			for j in range(i+1,no_0):
				T_1st_contrib += self.distance_squared(features_0,features_0,i,j)
		T_1st_contrib = T_1st_contrib/(no_0*(no_0-1))

                T_2nd_contrib=0

                for i in range(no_1):
                        for j in range(i+1,no_1):
                                T_2nd_contrib += self.distance_squared(features_1,features_1,i,j)
                T_2nd_contrib = T_2nd_contrib/(no_1*(no_1-1))


                T_3rd_contrib=0

		no_2=no_0+no_1

                for i in range(no_0):
                        for j in range(no_1):
                                T_3rd_contrib += self.distance_squared(features_0,features_1,i,j)
                T_3rd_contrib = T_3rd_contrib/(no_2*(no_2-1))

		T = T_1st_contrib + T_2nd_contrib +  T_3rd_contrib

                with open("test_statistic_distributions/test_statistics_"+self.name+"_"+self.sample1_name+"_"+self.sample2_name+"_energy_test_"+str(self.sigma), "a") as test_statistics_file:
                        test_statistics_file.write("{0} \t{1} \t{2} \t{3} \t{4} \n".format(0,0,0,0,T))



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
		
		self.saver = tf.train.Saver(tf.all_variables())
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
		correct_prediction_float=tf.cast(correct_prediction, "float")

		prediction = tf.argmax(y,1)
		
		if __debug__:
			print("correct_prediction_float")
			print(correct_prediction_float.eval(feed_dict={x: self.X_val[:,:].reshape(self.no_validation,2), y_: self.y_val_tf[:,:].reshape(self.no_validation,2)}))

			print("prediction")
			print(prediction.eval(feed_dict={x: self.X_val[:,:].reshape(self.no_validation,2)}))

			print("actual")
			print(self.y_val_tf)

			print("accuracy on validation sample")
			print(accuracy.eval(feed_dict={x: self.X_val[:,:].reshape(self.no_validation,2), y_: self.y_val_tf[:,:].reshape(self.no_validation,2)}))
			self.print_line()

		self.save_path = self.saver.save(sess, "/tmp/model.ckpt")
		if __debug__:print("Model saved in file: ", self.save_path)



	def predict(self, X_val):
		self.saver.restore(self.sess, "/tmp/model.ckpt")
		print("Model restored") 
		prediction = tf.argmax(y,1)
		return prediction.eval(feed_dict={x: self.X_val[:,:].reshape(self.no_validation,2)})


	def predict_proba(self, X_val):
		print("Need to implement the method PREDICT_PROBA")

class dt_sklearn(sklearn_classifier):
    clf_blueprint = tree.DecisionTreeClassifier()
    specific_type_of_classifier="dt"
    
class ada_sklearn(sklearn_classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0, no_estimators=1000,name="unnamed",sample1_name="sample1",sample2_name="sample2"):
        self.no_estimators = no_estimators
        if __debug__:print("no_estimators: %.4f" % self.no_estimators)
        self.clf_blueprint = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=no_estimators)
        self.specific_type_of_classifier="ada_"+str(self.no_estimators)+"estimators"
	super( ada_sklearn, self ).__init__(data,percentage_used_for_validation,no_permutations,name,sample1_name,sample2_name)
    def set_no_estimators(self,no_estimators):
        self.no_estimators=no_estimators
    def get_no_estimators(self):
        return self.no_estimators
    
class svm_sklearn(sklearn_classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0, a_cache_size=1000,name="unnamed",sample1_name="sample1",sample2_name="sample2"):
        self.cache_size = a_cache_size
        if __debug__:print("cache_size: %.4f" % self.cache_size)
        self.clf_blueprint = svm.SVC(probability=True,cache_size=a_cache_size)
        self.specific_type_of_classifier="svm"
	super( svm_sklearn, self ).__init__(data,percentage_used_for_validation,no_permutations,name,sample1_name,sample2_name)
    def set_cache_size(self,a_cache_size):
        self.cache_size=a_cache_size
    def get_cache_size(self):
        return self.cache_size
    
class nn_sklearn(sklearn_classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0,name="unnamed",sample1_name="sample1",sample2_name="sample2"):
        self.clf_blueprint = Classifier(
            layers=[
                Layer("Rectifier", units=10),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=1)
        self.specific_type_of_classifier="nn_rectifier10_softmax_lr0_001"
	super( nn_sklearn, self ).__init__(data,percentage_used_for_validation,no_permutations,name,sample1_name,sample2_name)






