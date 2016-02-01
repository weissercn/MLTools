import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
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
        
        self.X_val = self.data_validation[:,:-1]
        self.no_validation=self.X_val.shape[0]
        self.y_val = self.data_validation[:,2:].reshape((self.no_validation,))
        
        #Creating the classifier from blueprint. The blueprint is provided by the daughter class.
        self.clf = self.clf_blueprint
        print(self.clf)
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
        self.X_val=-1
        self.y_val=-1
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
    
    def train_from_scratch(self):
        self.clf = self.clf_blueprint
        self.clf = self.clf.fit(self.X_pri, self.y_pri)
        
    def train(self):
        self.clf = self.clf.fit(self.X_pri, self.y_pri)
        
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
        n0, bins0, patches0 = plt.hist(self.data_validation_file_0[:,4], bins=bins_probability, normed=True, facecolor='red', alpha=0.5, label="file_0")
        plt.xlabel('Probability')
        plt.ylabel('Normalised Frequency')
        plt.title('Probability Predictions Particle Sample BDT')
        plt.savefig('Machine_learning_predictions_particle_BDT.pdf', format='pdf', dpi=300)
        #plt.show()
        n1, bins1, patches1 = plt.hist(self.data_validation_file_1[:,4], bins=bins_probability, normed=True, facecolor='blue', alpha=0.5, label="file_1")
        #plt.axis([0, 1, 0, 0.03])
        plt.xlabel('Probability')
        plt.ylabel('Normalised Frequency')
        plt.title('Probability Predictions CPV2 Sample BDT')
        plt.savefig('Machine_learning_predictions_CPV2_BDT.pdf', format='pdf', dpi=300)
        #plt.show()
        
        
        n0, bins0, patches0 = plt.hist(self.data_validation_file_0[:,4], bins=bins_probability, normed=True, facecolor='red', alpha=0.5, label="Particle")
        n1, bins1, patches1 = plt.hist(self.data_validation_file_1[:,4], bins=bins_probability, normed=True, facecolor='blue', alpha=0.5, label="Antiparticle")
        #plt.axis([0.46, 0.53,0,600])
        plt.legend(loc='upper right')
        plt.xlabel('Probability')
        plt.ylabel('Normalised Frequency')
        plt.title('Probability Predictions Particle and CPV2 Samples BDT')
        plt.savefig('Machine_learning_predictions_particle_CPV2_BDT.pdf', format='pdf', dpi=3000)
        #plt.show()
        
        
        #Subtract histograms. This is assuming equal bin width
        plt.scatter(bins_probability[:-1]+(bins_probability[1]-bins_probability[0])/2,n1-n0,facecolors='blue', edgecolors='blue')
        plt.xlabel('Probability')
        plt.ylabel('Normalised Frequency (sample 1 - sample 2)')
        plt.title('Differtial Probability Predictions Particle and CPV2 Samples BDT')
        plt.savefig('Machine_learning_predictions_CPV2_minus_particle_BDT.pdf', format='pdf', dpi=3000)
        #plt.show()
        
        #val[numpy.logical_or.reduce([val[:,1] == 1])]
        colorMap = plt.get_cmap('Spectral')
        
        plt.scatter( self.data_validation_file_0[:,0],self.data_validation_file_0[:,1],10,self.data_validation_file_0[:,3],cmap=colorMap)
        plt.savefig('validation_file_0.pdf', format='pdf', dpi=300)
        
        print("pred_validation_file_0")
        print(pred_validation_file_0)
        print("pred_validation_file_1")
        print(pred_validation_file_1)
        result_KS=stats.ks_2samp(self.data_validation_file_0[:,4], self.data_validation_file_1[:,4])
        print(result_KS)
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
        

class odt(classifier):
    clf_blueprint = tree.DecisionTreeClassifier()
    
class obdt(classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0, no_estimators=1000):
        self.no_estimators = no_estimators
        print("no_estimators: %.4f" % self.no_estimators)
        self.clf_blueprint = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=no_estimators)
        super( obdt, self ).__init__(data,percentage_used_for_validation,no_permutations)
    def set_no_estimators(self,no_estimators):
        self.no_estimators=no_estimators
    def get_no_estimators(self):
        return self.no_estimators
    
class osvm(classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0, a_cache_size=1000):
        self.cache_size = a_cache_size
        print("cache_size: %.4f" % self.cache_size)
        self.clf_blueprint = svm.SVC(probability=True,cache_size=a_cache_size)
        super( osvm, self ).__init__(data,percentage_used_for_validation,no_permutations)
    def set_cache_size(self,a_cache_size):
        self.cache_size=a_cache_size
    def get_cache_size(self):
        return self.cache_size
    
class onn(classifier):
    def __init__(self,data,percentage_used_for_validation,no_permutations=0):
        self.clf_blueprint = Classifier(
            layers=[
                Layer("Rectifier", units=10),
                Layer("Softmax")],
            learning_rate=0.001,
            n_iter=1)
        print(self.clf_blueprint)
        super( onn, self ).__init__(data,percentage_used_for_validation,no_permutations)







