import sys
import numpy as np
import matplotlib.pyplot as plt
import os

dimensions=[1,2,3,4]

print("Legendre dimensional analysis \n")

p_bdt = []
for dim in range(1,5):
        temp = np.loadtxt("../bdt_legendre/"+str(dim)+'Dlegendre4contrib_bdt_p_values')
        p_bdt.append(temp)

print("Boosted decision tree : ", p_bdt)

p_svm = []
for dim in range(1,5):
        temp = np.loadtxt("../svm_legendre/"+str(dim)+'Dlegendre4contrib_svm_p_values')
        p_svm.append(temp)

print("Support vector machine : ", p_svm)


p_nn = []
for dim in range(1,5):
        temp = np.loadtxt("../nn_legendre/"+str(dim)+'Dlegendre4contrib_nn_4layers_100neurons_onehot_p_values')
        p_nn.append(temp)

print("Neural Network : ", p_nn)

p_miranda_3bins = [1.2212453270876722e-15, 0.012266775041445577,0.18759112444309145,0.29341559120156957]
#for dim in range(1,5):
        #temp = np.loadtxt(os.environ['MLToolsDir']+"/Dalitz_simplified/optimisation/miranda/legendre_4contrib_"+str(dim)+'D_optimisation_miranda_values')
        #p_miranda_3bins.append(temp)

print("Miranda 3 bins: ", p_miranda_3bins)

fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(dimensions,p_bdt,label="bdt ",color='darkorange')
ax.plot(dimensions,p_svm,label="svm ",color='lawngreen')

ax.plot(dimensions,p_nn,label="nn 4l 100n ",color='blueviolet')

ax.plot(dimensions,p_miranda_3bins,label="Miranda 3bins",color='darkred')

plt.ylim([0,1])
ax.set_xlabel("Number of dimensions")
ax.set_ylabel("P value")
ax.set_title("Dimensionality analysis legendre 4 contrib")
ax.legend(loc='upper right')
fig_name="legendre4contrib_dimensionality_analysis"
fig.savefig(fig_name)
fig.savefig("../bdt_legendre/"+fig_name)
fig.savefig("../svm_legendre/"+fig_name)
fig.savefig("../nn_legendre/"+fig_name)
print("Saved the figure as" , fig_name+".png")


