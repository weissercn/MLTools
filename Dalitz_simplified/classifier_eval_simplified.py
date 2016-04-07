
#adapted from the example at http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
"""
This script can be used to get the p value for classifiers. It takes input files with column vectors corresponding to features and lables. 
Then there are two different routes one can go down. When optimise_hyperparam_mode is enabled (1), then a grid search will be performed on 
one set of input files. When the mode is turned off (0), then the p value is computed for multiple sets of input files and the p value 
distribution is plotted.  
"""

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

def classifier_eval_simplified(aC,agamma):
	##############################################################################
	# Setting parameters
	#

	name="dalitz"
	sample1_name="particle"
	sample2_name="antiparticle"

	shuffling_seed = 100

	cv_n_iter = 5

	optimise_hyperparam_mode=0

	if optimise_hyperparam_mode==1:
		#For optimise_hyperparam_mode=1
		comp_file_0 = os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.0.0.txt"
		comp_file_1 = os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.200.1.txt"

		comp_file_list = [(comp_file_0, comp_file_1)]

		C_range = np.logspace(-2, 10, 13)
		gamma_range = np.logspace(-9, 3, 13)
		param_grid = dict(gamma=gamma_range, C=C_range)

	else:
		#For optimise_hyperparam_mode=0
		clf = SVC(C=aC,gamma=agamma,probability=True)
		comp_file_list=[]

		for i in range(1,100):
			comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.{0}.0.txt".format(i), os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data.2{0}.1.txt".format(str(i).zfill(2))))
		print(comp_file_list)
		score_list=[]


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


		cv = StratifiedShuffleSplit(y, n_iter=cv_n_iter, test_size=0.2, random_state=42)

		print(X)
		print(y)

		# It is usually a good idea to scale the data for SVM training.
		# We are cheating a bit in this example in scaling all of the data,
		# instead of fitting the transformation on the training set and
		# just applying it on the test set.

		scaler = StandardScaler()
		X = scaler.fit_transform(X)

		if optimise_hyperparam_mode==1:
			##############################################################################
			# Train classifiers
			#
			# For an initial search, a logarithmic grid with basis
			# 10 is often helpful. Using a basis of 2, a finer
			# tuning can be achieved but at a much higher cost.


			grid = GridSearchCV(SVC(probability=True), scoring=p_value_scoring_object.p_value_scoring_object ,param_grid=param_grid, cv=cv)
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
			scores = cross_validation.cross_val_score(clf,X,y,cv=cv_n_iter,scoring=p_value_scoring_object.p_value_scoring_object)	
			print(scores)
			score_list.append(np.mean(scores))


	############################################################################################################################################################
	###############################################################  Evaluation of results  ####################################################################
	############################################################################################################################################################


	if optimise_hyperparam_mode==0:
		# The score list has been computed. Let's plot the distribution
		print(score_list)



if __name__ == "__main__":
	print("Executing classifier_eval_simplified as a stand-alone script")
	print()
	#classifier_eval_simplified(aC,agamma)
	classifier_eval_simplified(1,0.1)
