from __future__ import division
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os

data=np.loadtxt("optimisation_values.txt",dtype='d')
x=data[:50,0]
y=data[:50,1]
z=data[:50,2]
cm = plt.cm.get_cmap('RdYlBu')

fig= plt.figure()
ax1= fig.add_subplot(1, 1, 1)

sc = ax1.scatter(x,y,c=z,s=35,cmap=cm)

print("z : ",z)
index = np.argmin(z)
print("index of max : ",index)
print("values of max : ",x[index],y[index],z[index])
ax1.scatter(x[index],y[index],c=z[index],s=50,cmap=cm)


cb=plt.colorbar(sc)

cb.set_label('p value')
ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_title('optimisation of hyperparameters for svm 4D Gauss')
fig.savefig("svm_optimisation_values_4DGauss.png")

