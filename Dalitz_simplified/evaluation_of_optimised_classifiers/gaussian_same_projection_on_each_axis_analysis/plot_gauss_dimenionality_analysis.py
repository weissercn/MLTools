import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Options for mode 'single_p_values','ensemble', 'ensemble_redefined', 'ensemble_redefined_noCPV', 'ensemble_redefined_optimised', 'ensemble_redefined_noCPV_optimised'
MODE= 'ensemble_redefined_noCPV_optimised'

if MODE == 'single_p_values':
	dimensions=[2,3,4,5,6,7,8,9,10]

	print("Gaussian same projection on each axis dimensional analysis \n")
	
	p_bdt = []
	for dim in dimensions:
		temp = np.loadtxt("../bdt_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection__0_1__0_085_bdt_p_values')
		p_bdt.append(temp)

	print("Boosted decision tree : ", p_bdt)

	p_svm = []
	for dim in dimensions:
		temp = np.loadtxt("../svm_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection__0_1__0_085_svm_p_values')
		p_svm.append(temp)

	print("Support vector machine : ", p_svm)


	p_nn = []
	for dim in dimensions:
		temp = np.loadtxt("../nn_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection__0_1__0_085_nn_4layers_100neurons_onehot_p_values')
		p_nn.append(temp)

	print("Neural Network : ", p_nn)

	p_miranda_2bins = []
	for dim in dimensions:
		temp = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_p_value_distribution__0_1__0_085_CPV_miranda_"+ str(dim)+ "D_2_bins_p_values")
		p_miranda_2bins.append(temp)
	print("Miranda 2 bins : ",p_miranda_2bins )

	p_miranda_3bins = [ ]
	for dim in dimensions:
		temp = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_p_value_distribution__0_1__0_085_CPV_miranda_"+ str(dim)+ "D_3_bins_p_values")
		p_miranda_3bins.append(temp)
	print("Miranda 3 bins : ",p_miranda_3bins )

	p_miranda_5bins = [ ]
	for dim in dimensions:
		temp = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_p_value_distribution__0_1__0_085_CPV_miranda_"+ str(dim)+ "D_5_bins_p_values")
		p_miranda_5bins.append(temp)
	print("Miranda 5 bins : ",p_miranda_5bins )



	fig = plt.figure()
	ax  = fig.add_subplot(1,1,1)
	ax.plot(dimensions,p_bdt,label="bdt ",color='darkorange')
	ax.plot(dimensions,p_svm,label="svm ",color='lawngreen')

	ax.plot(dimensions,p_nn,label="nn 4l 100n ",color='blueviolet')

	ax.plot(dimensions,p_miranda_2bins,label="Miranda 2bins",color='red')
	ax.plot(dimensions,p_miranda_3bins,label="Miranda 3bins",color='indianred')
	ax.plot(dimensions,p_miranda_5bins,label="Miranda 5bins",color='saddlebrown')

	ax.set_yscale('log')
	plt.ylim([0,1])
	ax.set_xlabel("Number of dimensions")
	ax.set_ylabel("P value")
	ax.set_title("Dimensionality analysis gaussian same projection sigmas perp .1 and 0.085")
	ax.legend(loc='lower left')
	fig_name="gaussian_same_projection__0_1__0_085_dimensionality_analysis"
	fig.savefig(fig_name)
	fig.savefig("../bdt_gaussian_same_projection/"+fig_name)
	fig.savefig("../svm_gaussian_same_projection/"+fig_name)
	fig.savefig("../nn_gaussian_same_projection/"+fig_name)
	fig.savefig("../miranda_gaussian_same_projection/"+fig_name)
	print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble':
	dimensions=[2,3,4,5,6,7,8,9,10]

	p_1_bdt = []
	p_2_bdt = []
	p_3_bdt = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../bdt_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection__0_1__0_085_bdt_p_values_1_2_3_std_dev.txt')
		p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

	print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

	p_1_svm = []
	p_2_svm = []
	p_3_svm = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../svm_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection__0_1__0_085_svm_ensemble_p_values_1_2_3_std_dev.txt')
		p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

	print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

	p_1_miranda_2bins = []
	p_2_miranda_2bins = []
	p_3_miranda_2bins = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_p_value_distribution__0_1__0_085_CPV_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
		p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

	print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

	p_1_miranda_3bins = []
	p_2_miranda_3bins = []
	p_3_miranda_3bins = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_p_value_distribution__0_1__0_085_CPV_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
		p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

	print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

	p_1_miranda_5bins = []
	p_2_miranda_5bins = []
	p_3_miranda_5bins = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_p_value_distribution__0_1__0_085_CPV_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
		p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

	print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)


	fig = plt.figure()
	ax  = fig.add_subplot(1,1,1)
	ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
	ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')

	ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
	ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
	ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')

	plt.ylim([-5,105])
	ax.set_xlabel("Number of dimensions")
	ax.set_ylabel("Number of samples")
	ax.set_title("Dimensionality analysis")
	ax.legend(loc='right')
	fig_name="gaussian_same_projection__0_1__0_085_ensemble_dimensionality_analysis"
	fig.savefig(fig_name)
	fig.savefig("../bdt_gaussian_same_projection/"+fig_name)
	fig.savefig("../svm_gaussian_same_projection/"+fig_name)
	fig.savefig("../miranda_gaussian_same_projection/"+fig_name)
	print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_redefined':
	dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_075_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

        p_1_svm = []
        p_2_svm = []
        p_3_svm = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../svm_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_075_svm_p_values_1_2_3_std_dev.txt')
                p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

        print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_075_CPV_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11): 
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_075_CPV_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_075_CPV_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

        print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)


        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
        ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')

        ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
        ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
        ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')

	plt.ylim([-5,120])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis redefined 0.075")
        ax.legend(loc='upper left')
        fig_name="gaussian_same_projection_redefined__0_1__0_075_ensemble_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_gaussian_same_projection/"+fig_name)
        fig.savefig("../svm_gaussian_same_projection/"+fig_name)
        fig.savefig("../miranda_gaussian_same_projection/"+fig_name)
        print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_redefined_noCPV': 
        dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_1_noCPV_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

        p_1_svm = []
        p_2_svm = []
        p_3_svm = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../svm_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_1_noCPV_svm_p_values_1_2_3_std_dev.txt')
                p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

        print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_1_noCPV_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_1_noCPV_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_1_noCPV_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

        print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)


        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
        ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')

	ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
        ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
        ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')

        plt.ylim([-5,105])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis redefined noCPV")
        ax.legend(loc='right')
        fig_name="gaussian_same_projection_redefined__0_1__0_1_noCPV_ensemble_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_gaussian_same_projection/"+fig_name)
        fig.savefig("../svm_gaussian_same_projection/"+fig_name)
        fig.savefig("../miranda_gaussian_same_projection/"+fig_name)
        print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_redefined_optimised':
        dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_075_optimised_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

	p_1_svm = []
	p_2_svm = []
	p_3_svm = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../svm_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_075_optimised_svm_p_values_1_2_3_std_dev.txt')
		p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

	print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

	p_1_nn = []
        p_2_nn = []
        p_3_nn = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../nn_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_075_optimised_p_values_1_2_3_std_dev.txt')
                p_1_nn.append(temp1), p_2_nn.append(temp2), p_3_nn.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_075_CPV_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_075_CPV_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_075_CPV_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

        print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)


        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
        ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')
	ax.plot(dimensions,p_2_nn,label="nn 2$\sigma$",color='blue')

        ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
        ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
        ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')

        plt.ylim([-5,120])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis redefined 0.075")
        ax.legend(loc='best')
        fig_name="gaussian_same_projection_redefined__0_1__0_075_optimised_ensemble_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_gaussian_same_projection/"+fig_name)
        fig.savefig("../svm_gaussian_same_projection/"+fig_name)
        fig.savefig("../miranda_gaussian_same_projection/"+fig_name)
        print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_redefined_noCPV_optimised': 
        dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_1_noCPV_optimised_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

	p_1_svm = []
	p_2_svm = []
	p_3_svm = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../svm_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_1_noCPV_optimised_svm_p_values_1_2_3_std_dev.txt')
		p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

        print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

	p_1_nn = []
        p_2_nn = []
        p_3_nn = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../nn_gaussian_same_projection/"+str(dim)+'Dgaussian_same_projection_redefined__0_1__0_1_noCPV_optimised_p_values_1_2_3_std_dev.txt')
                p_1_nn.append(temp1), p_2_nn.append(temp2), p_3_nn.append(temp3)

        print("Neural Network 3 layers with 33 neurons : ", p_1_nn,p_2_nn,p_3_nn)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_1_noCPV_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_1_noCPV_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_gaussian_same_projection/gaussian_same_projection_redefined_p_value_distribution__0_1__0_1_noCPV_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

        print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)


        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
        ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')
	ax.plot(dimensions,p_2_nn,label="nn 2$\sigma$",color='blue')

        ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
        ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
        ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')

        plt.ylim([-5,105])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis redefined noCPV")
        ax.legend(loc='right')
        fig_name="gaussian_same_projection_redefined__0_1__0_1_noCPV_optimised_ensemble_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_gaussian_same_projection/"+fig_name)
        fig.savefig("../svm_gaussian_same_projection/"+fig_name)
        fig.savefig("../miranda_gaussian_same_projection/"+fig_name)
        print("Saved the figure as" , fig_name+".png")


else:
	print("No valid mode entered")



