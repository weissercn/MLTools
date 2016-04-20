
#adapted from the example at http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
"""
This script can be used to get the p value for classifiers. It takes input files with column vectors corresponding to features and lables. 
Then there are two different routes one can go down. When mode has a value of 1, then a grid search will be performed on 
one set of input files. If it is 2, then the hyperparemeter search is performed by spearmint. When the mode is turned off (0), 
then the p value is computed for multiple sets of input files and the p value distribution is plotted. One sets all the valiables 
including the classifier in the "args" list. The classifier provided is ignored if keras_mode is on (1) in which case a keras neural 
network is used.   
"""

from __future__ import print_function
print(__doc__)
import os
import p_value_scoring_object
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from scipy import stats
import math

##############################################################################
# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) 

class Counter(object):
    # Creating a counter object to be able to perform cross validation with only one split
    def __init__(self, list1,list2):
        self.current = 1 
        self.list1 =list1
        self.list2 =list2

    def __iter__(self):
        'Returns itself as an iterator object'
        return self

    def __next__(self):
        'Returns the next value till current is lower than high'
        if self.current > 1:
            raise StopIteration
        else:
            self.current += 1
            return self.list1,self.list2 
    next = __next__ #python2 

def histo_plot_pvalue(U_0,abins,axlabel,aylabel,atitle,aname):
        bins_probability=np.histogram(U_0,bins=abins)[1]

        #Finding the p values corresponding to 1,2 and 3 sigma significance.
        no_one_std_dev=sum(i < (1-0.6827) for i in U_0) 
        no_two_std_dev=sum(i < (1-0.9545) for i in U_0)
        no_three_std_dev=sum(i < (1-0.9973) for i in U_0)

        print(no_one_std_dev,no_two_std_dev,no_three_std_dev)

        #plt.rc('text', usetex=True)
        textstr = '$1\sigma=%i$\n$2\sigma=%i$\n$3\sigma=%i$'%(no_one_std_dev, no_two_std_dev, no_three_std_dev)


        # Making a histogram of the probability predictions of the algorithm. 
        fig_pred_0= plt.figure()
        ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
        n0, bins0, patches0 = ax1_pred_0.hist(U_0, bins=bins_probability, facecolor='red', alpha=0.5)
        ax1_pred_0.set_xlabel(axlabel)
        ax1_pred_0.set_ylabel(aylabel)
        ax1_pred_0.set_title(atitle)
        plt.xlim([0,1])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax1_pred_0.text(0.85, 0.95, textstr, transform=ax1_pred_0.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

        fig_pred_0.savefig(aname+".png")
        #fig_pred_0.show()
        plt.close(fig_pred_0) 


def classifier_eval(mode,keras_mode,args):
	##############################################################################
	# Setting parameters
	#


	name=args[0]
	sample1_name= args[1]
	sample2_name= args[2]

	shuffling_seed = args[3]

	#mode =0 if you want evaluation of a model =1 if grid hyperparameter search =2 if spearmint hyperparameter search
	comp_file_list=args[4]
	print(comp_file_list)
	cv_n_iter = args[5]
	clf = args[6]
	C_range = args[7]
	gamma_range = args[8]

        if mode==0:
                #For standard evaluation
                score_list=[]
		print("standard evaluation mode")
	elif mode==1:
		#For grid search
		print("grid hyperparameter search mode")
		param_grid = dict(gamma=gamma_range, C=C_range)

	elif mode==2:
		#For spearmint hyperparameter search
		print("spearmint hyperparameter search mode")
	else:
		print("No valid mode chosen")
		return 1
	

	##############################################################################
	# Load and prepare data set
	#
	# dataset for grid search

	for comp_file_0,comp_file_1 in comp_file_list:

		print("Operating of files :"+comp_file_0+"   "+comp_file_1)

		#extracts data from the files
		features_0=np.loadtxt(comp_file_0,dtype='d')
		features_1=np.loadtxt(comp_file_1,dtype='d')

		#determine how many data points are in each sample
		no_0=features_0.shape[0]
		no_1=features_1.shape[0]
		no_tot=no_0+no_1
		#Give all samples in file 0 the label 0 and in file 1 the feature 1
		label_0=np.zeros((no_0,1))
		label_1=np.ones((no_1,1))

		#Create an array containing samples and features.
		data_0=np.c_[features_0,label_0]
		data_1=np.c_[features_1,label_1]

		data=np.r_[data_0,data_1]

		np.random.shuffle(data)

		X=data[:,:-1]
		y=data[:,-1]
		print("X : ",X)
		print("y : ",y)
		atest_size=0.2
		if cv_n_iter==1:
			train_range = range(int(math.floor(no_tot*(1-atest_size))))
			test_range  = range(int(math.ceil(no_tot*(1-atest_size))),no_tot)
			#print("train_range : ", train_range)
			#print("test_range : ", test_range)
			acv = Counter(train_range,test_range)
			#print(acv)
		else:
			acv = StratifiedShuffleSplit(y, n_iter=cv_n_iter, test_size=atest_size, random_state=42)

		print("Finished with setting up samples")

		# It is usually a good idea to scale the data for SVM training.
		# We are cheating a bit in this example in scaling all of the data,
		# instead of fitting the transformation on the training set and
		# just applying it on the test set.

		scaler = StandardScaler()
		X = scaler.fit_transform(X)

		if mode==1:
			##############################################################################
			# Grid Search
			#
			# Train classifiers
			#
			# For an initial search, a logarithmic grid with basis
			# 10 is often helpful. Using a basis of 2, a finer
			# tuning can be achieved but at a much higher cost.


			grid = GridSearchCV(clf, scoring=p_value_scoring_object.p_value_scoring_object ,param_grid=param_grid, cv=acv)
			grid.fit(X, y)

			print("The best parameters are %s with a score of %0.2f"
					% (grid.best_params_, grid.best_score_))

			# Now we need to fit a classifier for all parameters in the 2d version
			# (we use a smaller set of parameters here because it takes a while to train)

			C_2d_range = [1e-2, 1, 1e2]
			gamma_2d_range = [1e-1, 1, 1e1]
			classifiers = []
			for C in C_2d_range:
				for gamma in gamma_2d_range:
					clf = SVC(C=C, gamma=gamma)
					clf.fit(X_2d, y_2d)
					classifiers.append((C, gamma, clf))

			##############################################################################
			# visualization
			#
			# draw visualization of parameter effects

			plt.figure(figsize=(8, 6))
			xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
			for (k, (C, gamma, clf)) in enumerate(classifiers):
				# evaluate decision function in a grid
				Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
				Z = Z.reshape(xx.shape)

				# visualize decision function for these parameters
				plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
				plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),size='medium')

				# visualize parameter's effect on decision function
				plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
				plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
				plt.xticks(())
				plt.yticks(())
				plt.axis('tight')

				plt.savefig('prediction_comparison.png')
				# plot the scores of the grid
				# grid_scores_ contains parameter settings and scores
				# We extract just the scores
				scores = [x[1] for x in grid.grid_scores_]
				scores = np.array(scores).reshape(len(C_range), len(gamma_range))

			# Draw heatmap of the validation accuracy as a function of gamma and C
			#
			# The score are encoded as colors with the hot colormap which varies from dark
			# red to bright yellow. As the most interesting scores are all located in the
			# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
			# as to make it easier to visualize the small variations of score values in the
			# interesting range while not brutally collapsing all the low score values to
			# the same color.

			plt.figure(figsize=(8, 6))
			plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
			plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
					norm=MidpointNormalize(vmin=-1.0, midpoint=-0.0001))
			plt.xlabel('gamma')
			plt.ylabel('C')
			plt.colorbar()
			plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
			plt.yticks(np.arange(len(C_range)), C_range)
			plt.title('Validation accuracy')
			plt.savefig('Heat_map.png')
		else:
			if keras_mode==1:
				from keras.models import Sequential
				from keras.layers.core import Dense, Activation
				from keras.layers import Dropout
				from keras.utils import np_utils, generic_utils

				dimof_input = X.shape[1]
				dimof_output =1
				
				print("dimof_input : ",dimof_input, "dimof_output : ", dimof_output)				
				#y = np_utils.to_categorical(y, dimof_output)
				scores = []
				counter = 1
				for train_index, test_index in acv:
					print("Cross validation run ", counter)
					X_train, X_test = X[train_index], X[test_index]
					y_train, y_test = y[train_index], y[test_index]
					
					print("X_train : ",X_train)
					print("y_train : ",y_train)

					batch_size = 1 
					dimof_middle = 100 
					dropout = 0.5 
					countof_epoch = 5 
					n_hidden_layers =4

					model = Sequential() 
					model.add(Dense(input_dim=dimof_input, output_dim=dimof_middle, init="glorot_uniform",activation='tanh'))
					model.add(Dropout(dropout))

					for n in range(n_hidden_layers):
						model.add(Dense(input_dim=dimof_middle, output_dim=dimof_middle, init="glorot_uniform",activation='tanh'))
						model.add(Dropout(dropout))
							
					model.add(Dense(input_dim=dimof_middle, output_dim=dimof_output, init="glorot_uniform",activation='sigmoid'))

					print("Does it make it here?")
					#Compiling (might take longer)
					model.compile(loss='categorical_crossentropy', optimizer='sgd')
					model.fit(X_train, y_train,show_accuracy=True,batch_size=batch_size, nb_epoch=countof_epoch, verbose=0)
					prob_pred = model.predict_proba(X_test)
					print("prob_pred : ", prob_pred)
					assert (not (np.isnan(np.sum(prob_pred))))
				
					#Just like in p_value_scoring_strategy.py
				        y_test         = np.reshape(y_test,(1,y_test.shape[0]))
					prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))
					prob_0    = prob_pred[np.logical_or.reduce([y_test==0])]
					prob_1    = prob_pred[np.logical_or.reduce([y_test==1])]
					if __debug__:
						print("Plot")
					p_KS=stats.ks_2samp(prob_0,prob_1)
					print(p_KS)
					scores.append(p_KS[1])
					counter +=1
	
					
			else:
				scores = cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.p_value_scoring_object)	
			print(scores)
			score_list.append(np.mean(scores))
			if mode==2:
				return (-1)* np.mean(scores)

	############################################################################################################################################################
	###############################################################  Evaluation of results  ####################################################################
	############################################################################################################################################################


	if mode==0:
		# The score list has been computed. Let's plot the distribution
		print(score_list)
		print("I havent implemented plotting of the distribution")
		histo_plot_pvalue(score_list,50,"p value","Frequency","p value distribution",name[i]+"_p_value")

if __name__ == "__main__":
	print("Executing classifier_eval_simplified as a stand-alone script")
	print()
        comp_file_list=[]
	for i in range(1,10):
		comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.{0}.0.txt".format(i), os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.2{0}.1.txt".format(str(i).zfill(2))))

	clf = SVC(C=100,gamma=0.1,probability=True, cache_size=7000)
	args=["dalitz_svc","particle","antiparticle",100,comp_file_list,1,clf,np.logspace(-2, 10, 13),np.logspace(-9, 3, 13)]
	#classifier_eval_simplified(aC,agamma)
	classifier_eval(0,0,args)
