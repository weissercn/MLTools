import sys
import numpy as np
import matplotlib.pyplot as plt

dimensions=[2,3,4,5,6,7,8,9,10]

#Decision Tree
p_1_dt = []
p_2_dt = []
p_3_dt = []
for dim in range(2,11):
	temp1,temp2,temp3 = np.loadtxt("../dt_gauss/"+str(dim)+'Dgauss_dt_p_values_1_2_3_std_dev.txt')
	p_1_dt.append(temp1), p_2_dt.append(temp2), p_3_dt.append(temp3)

print("Decision tree : ", p_1_dt,p_2_dt,p_3_dt)

p_1_bdt = []
p_2_bdt = []
p_3_bdt = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../bdt_gauss/"+str(dim)+'Dgauss_bdt_p_values_1_2_3_std_dev.txt')
        p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

p_1_bdt_AD = []
p_2_bdt_AD = []
p_3_bdt_AD = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../bdt_gauss/"+str(dim)+'Dgauss_bdt_AD_p_values_1_2_3_std_dev.txt')
        p_1_bdt_AD.append(temp1), p_2_bdt_AD.append(temp2), p_3_bdt_AD.append(temp3)

print("Boosted decision tree Anderson Darling : ", p_1_bdt_AD,p_2_bdt_AD,p_3_bdt_AD)

p_1_svm = []
p_2_svm = []
p_3_svm = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../svm_gauss/"+str(dim)+'Dgauss_svm_p_values_1_2_3_std_dev.txt')
        p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

p_1_nn_6_200 = []
p_2_nn_6_200 = []
p_3_nn_6_200 = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../nn_gauss/"+str(dim)+'Dgauss_nn_p_values_1_2_3_std_dev.txt')
        p_1_nn_6_200.append(temp1), p_2_nn_6_200.append(temp2), p_3_nn_6_200.append(temp3)

print("Neural Network 6 layers 200 neurons : ", p_1_nn_6_200,p_2_nn_6_200,p_3_nn_6_200)

p_1_nn_4_100 = []
p_2_nn_4_100 = []
p_3_nn_4_100 = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../nn_gauss/"+str(dim)+'Dgauss_nn_4layers_100neurons_p_values_1_2_3_std_dev.txt')
        p_1_nn_4_100.append(temp1), p_2_nn_4_100.append(temp2), p_3_nn_4_100.append(temp3)

print("Neural Network 4 layers 100 neurons : ", p_1_nn_4_100,p_2_nn_4_100,p_3_nn_4_100)


# Using the old architecture
dim= np.array([10,2,3,4,5,6,7,8,9])
p = dim.argsort()
p_nn_4_100_total  = np.loadtxt('gaussian_dimensionality_analysis_nn')
p_1_nn_4_100_old = p_nn_4_100_total[p,0].tolist()
p_2_nn_4_100_old = p_nn_4_100_total[p,1].tolist()
p_3_nn_4_100_old = p_nn_4_100_total[p,2].tolist()
print("Neural Network 4 layers 100 neurons old architecture : ", p_1_nn_4_100_old,p_2_nn_4_100_old,p_3_nn_4_100_old)


p_1_miranda_2bins = []
p_2_miranda_2bins = []
p_3_miranda_2bins = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../miranda_gauss/"+str(dim)+'Dgauss_miranda_2bins_p_values_1_2_3_std_dev.txt')
        p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

p_1_miranda_3bins = []
p_2_miranda_3bins = []
p_3_miranda_3bins = []
for dim in range(2,11):
        temp1,temp2,temp3 = np.loadtxt("../miranda_gauss/"+str(dim)+'Dgauss_miranda_3bins_p_values_1_2_3_std_dev.txt')
        p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(dimensions,p_2_dt,label="dt 2$\sigma$",color='black')
ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
#ax.plot(dimensions,p_2_bdt_AD,label="bdt AD 2$\sigma$",color='saddlebrown')
ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')

ax.plot(dimensions,p_2_nn_6_200,label="nn 6l 200n 2$\sigma$",color='blue')
ax.plot(dimensions,p_2_nn_4_100,label="nn 4l 100n 2$\sigma$",color='blueviolet')
ax.plot(dimensions,p_2_nn_4_100_old,label="nn 4l 100n old 2$\sigma$",color='cyan')

ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='darkred')

plt.ylim([0,100])
ax.set_xlabel("Number of dimensions")
ax.set_ylabel("Number of samples")
ax.set_title("Dimensionality analysis")
ax.legend(loc='upper right')
fig_name="dimensionality_analysis"
fig.savefig(fig_name)
fig.savefig("../dt_gauss/"+fig_name)
fig.savefig("../bdt_gauss/"+fig_name)
fig.savefig("../svm_gauss/"+fig_name)
fig.savefig("../nn_gauss/"+fig_name)
fig.savefig("../miranda_gauss/"+fig_name)
print("Saved the figure as" , fig_name+".png")


