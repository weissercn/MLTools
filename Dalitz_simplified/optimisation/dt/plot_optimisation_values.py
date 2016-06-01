from __future__ import division
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os

class_name = "dt"
name = "Dalitz"
avmin=0.01

filename= class_name+"_optimisation_values_"+name
data=np.loadtxt(filename+".txt",dtype='d')
x=data[:50,0]
y=data[:50,1]
z=data[:50,2]
cm = plt.cm.get_cmap('RdYlBu')

fig= plt.figure()
ax1= fig.add_subplot(1, 1, 1)

sc = ax1.scatter(x,y,c=z,s=35,cmap=cm, norm=colors.LogNorm(),vmin=avmin,vmax=1)

print("z : ",z)
index = np.argmin(z)
print("index of max : ",index)
print("values of max : ",x[index],y[index],z[index])
ax1.scatter(x[index],y[index],c=z[index], norm=colors.LogNorm(),s=50, cmap=cm,vmin=avmin,vmax=1)

cb=fig.colorbar(sc,ticks=[1,0.1,0.01,0.001])
cb.ax.set_yticklabels(['1','0.1','0.01','0.001'])
cb.set_label('p value')

ax1.set_xlabel('C')
ax1.set_ylabel('gamma')
ax1.set_title('optimisation of hyperparameters for '+class_name+' '+name)
print("saving to "+filename+".png")
fig.savefig(filename+".png")

