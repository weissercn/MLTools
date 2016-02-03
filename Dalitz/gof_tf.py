# source activate python27environment
# you need to type this line into the terminal to enable the conda environement in which python 2.7 is installed.
# source deactivate
from __future__ import division
import matplotlib.pyplot as plt 
import numpy as np
import math
#from Users/weisser/anaconda/envs/python27environment/lib/python2.7/site-packages/tensorflow/models/image
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#file 0 contains the particle, file 1 the antiparticle samples.
comp_file_0='data.+.txt'
comp_file_1='data.cpv.v2.txt'

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

#USING STANDARD SCALER TO REMOVE MEAN AND STANDARD DEVIATION
#data[:,:-1]=preprocessing.StandardScaler().fit_transform(data[:,:-1])
#This should be done within the class

np.savetxt('data_unshuffled.txt', data)
#Shuffle data
np.random.shuffle(data)

np.savetxt('data.txt', data)

percentage_used_for_validation=40

#Defining primary and validation data
data_primary=data[:math.floor(data.shape[0]*(1-percentage_used_for_validation/100)),:]
data_validation=data[math.floor(data.shape[0]*(1-percentage_used_for_validation/100)):,:]

#Selecting features (X) and labels (y)
X_pri = data_primary[:,:-1]
no_primary=X_pri.shape[0]
y_pri = data_primary[:,2:].reshape((no_primary,))
y_pri_tf = np.zeros(( no_primary,2))

#print(y_pri)
#print(y_pri_tf)
for i in range(no_primary):
    if y_pri[i]==1:
	y_pri_tf[i,1]=1
    else:
	y_pri_tf[i,0]=1

#print(y_pri_tf)

X_val = data_validation[:,:-1]
no_validation=X_val.shape[0]
y_val = data_validation[:,2:].reshape((no_validation,))
y_val_tf = np.zeros(( no_validation,2))

for i in range(no_validation):
    if y_val[i]==1:
        y_val_tf[i,1]=1
    else:
        y_val_tf[i,0]=1

#print("test1")
import tensorflow as tf
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

for i in range(no_primary):
    #print(X_pri[i,:].reshape(1,2).shape)
    #print(y_pri_tf[i,:].reshape(1,2).shape)
    train_step.run(feed_dict={x: X_pri[i,:].reshape(1,2), y_: y_pri_tf[i,:].reshape(1,2)})

#print("test4")    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print("test5")
print("accuracy on validation sample")
print(accuracy.eval(feed_dict={x: X_val[:,:].reshape(no_validation,2), y_: y_val_tf[:,:].reshape(no_validation,2)}))

