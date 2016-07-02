import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Options for mode 'ensemble', 'ensemble_noCPV', 'ensemble_optimised', 'ensemble_noCPV_optimised'
MODE= 'ensemble_noCPV_optimised'

if MODE == 'ensemble':
	dimensions=[2,3,4,5,6,7,8,9,10]

	p_1_bdt = []
	p_2_bdt = []
	p_3_bdt = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../bdt_sin/"+str(dim)+'Dsin1diff_2_and_5_CPV_bdt_p_values_1_2_3_std_dev.txt')
		p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

	print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

	p_1_svm = []
	p_2_svm = []
	p_3_svm = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../svm_sin/"+str(dim)+'Dsin1diff_2_and_5_CPV_svm_p_values_1_2_3_std_dev.txt')
		p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

	print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

	p_1_miranda_2bins = []
	p_2_miranda_2bins = []
	p_3_miranda_2bins = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_5_periods_CPV_p_value_distribution_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
		p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

	print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

	p_1_miranda_3bins = []
	p_2_miranda_3bins = []
	p_3_miranda_3bins = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_5_periods_CPV_p_value_distribution_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
		p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

	print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_4bins = []
        p_2_miranda_4bins = []
        p_3_miranda_4bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_5_periods_CPV_p_value_distribution_miranda_"+str(dim)+'D_4_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_4bins.append(temp1), p_2_miranda_4bins.append(temp2), p_3_miranda_4bins.append(temp3)

        print("Miranda 4 bins: ", p_1_miranda_4bins,p_2_miranda_4bins,p_3_miranda_4bins)

	p_1_miranda_5bins = []
	p_2_miranda_5bins = []
	p_3_miranda_5bins = []
	for dim in range(2,11):
		temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_5_periods_CPV_p_value_distribution_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
		p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

	print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)

        p_1_miranda_6bins = []
        p_2_miranda_6bins = []
        p_3_miranda_6bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_5_periods_CPV_p_value_distribution_miranda_"+str(dim)+'D_6_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_6bins.append(temp1), p_2_miranda_6bins.append(temp2), p_3_miranda_6bins.append(temp3)

        print("Miranda 6 bins: ", p_1_miranda_6bins,p_2_miranda_6bins,p_3_miranda_6bins)

        #p_1_miranda_7bins = []
        #p_2_miranda_7bins = []
        #p_3_miranda_7bins = []
        #for dim in range(2,11):
        #        temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_5_periods_p_value_distribution_miranda_"+str(dim)+'D_7_bins_p_values_1_2_3_std_dev.txt')
        #        p_1_miranda_7bins.append(temp1), p_2_miranda_7bins.append(temp2), p_3_miranda_7bins.append(temp3)

        #print("Miranda 7 bins: ", p_1_miranda_7bins,p_2_miranda_7bins,p_3_miranda_7bins)

        #p_1_miranda_8bins = []
        #p_2_miranda_8bins = []
        #p_3_miranda_8bins = []
        #for dim in range(2,11):
                #temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_6_periods_p_value_distribution_miranda_"+str(dim)+'D_8_bins_p_values_1_2_3_std_dev.txt')
                #p_1_miranda_8bins.append(temp1), p_2_miranda_8bins.append(temp2), p_3_miranda_8bins.append(temp3)

        #print("Miranda 8 bins: ", p_1_miranda_8bins,p_2_miranda_8bins,p_3_miranda_8bins)

	fig = plt.figure()
	ax  = fig.add_subplot(1,1,1)
	ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
	ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')

	ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
	ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
	ax.plot(dimensions,p_2_miranda_4bins,label="Miranda 4bins 2$\sigma$",color='darkorchid')
	ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')
	ax.plot(dimensions,p_2_miranda_6bins,label="Miranda 6bins 2$\sigma$",color='deeppink')
        #ax.plot(dimensions,p_2_miranda_7bins,label="Miranda 7bins 2$\sigma$",color='darkorchid')
        #ax.plot(dimensions,p_2_miranda_8bins,label="Miranda 8bins 2$\sigma$",color='mediumpurple')

	plt.ylim([-5,105])
	ax.set_xlabel("Number of dimensions")
	ax.set_ylabel("Number of samples")
	ax.set_title("Dimensionality analysis sin")
	ax.legend(loc='right')
	fig_name="sin1diff_2_and_5_periods_dimensionality_analysis"
	fig.savefig(fig_name)
	fig.savefig("../bdt_sin/"+fig_name)
	fig.savefig("../svm_sin/"+fig_name)
	fig.savefig("../miranda_sin/"+fig_name)
	print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_noCPV':
        dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_sin/"+str(dim)+'Dsin1diff_2_and_2_noCPV_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

        p_1_svm = []
        p_2_svm = []
        p_3_svm = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../svm_sin/"+str(dim)+'Dsin1diff_2_and_2_noCPV_svm_p_values_1_2_3_std_dev.txt')
                p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

        print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_2_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_2_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

	p_1_miranda_4bins = []
        p_2_miranda_4bins = []
        p_3_miranda_4bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_2_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_4_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_4bins.append(temp1), p_2_miranda_4bins.append(temp2), p_3_miranda_4bins.append(temp3)

        print("Miranda 4 bins: ", p_1_miranda_4bins,p_2_miranda_4bins,p_3_miranda_4bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_2_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_5bins.append(temp1), p_2_miranda_5bins.append(temp2), p_3_miranda_5bins.append(temp3)

        print("Miranda 5 bins: ", p_1_miranda_5bins,p_2_miranda_5bins,p_3_miranda_5bins)

	p_1_miranda_6bins = []
        p_2_miranda_6bins = []
        p_3_miranda_6bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_2_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_6_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_6bins.append(temp1), p_2_miranda_6bins.append(temp2), p_3_miranda_6bins.append(temp3)

        print("Miranda 6 bins: ", p_1_miranda_6bins,p_2_miranda_6bins,p_3_miranda_6bins)

        #p_1_miranda_7bins = []
        #p_2_miranda_7bins = []
        #p_3_miranda_7bins = []
        #for dim in range(2,11):
        #        temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_2_and_2_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_7_bins_p_values_1_2_3_std_dev.txt')
        #        p_1_miranda_7bins.append(temp1), p_2_miranda_7bins.append(temp2), p_3_miranda_7bins.append(temp3)

        #print("Miranda 7 bins: ", p_1_miranda_7bins,p_2_miranda_7bins,p_3_miranda_7bins)

        #p_1_miranda_8bins = []
        #p_2_miranda_8bins = []
        #p_3_miranda_8bins = []
        #for dim in range(2,11):
                #temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_5_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_8_bins_p_values_1_2_3_std_dev.txt')
                #p_1_miranda_8bins.append(temp1), p_2_miranda_8bins.append(temp2), p_3_miranda_8bins.append(temp3)

        #print("Miranda 8 bins: ", p_1_miranda_8bins,p_2_miranda_8bins,p_3_miranda_8bins)

        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        ax.plot(dimensions,p_2_bdt,label="bdt 2$\sigma$",color='darkorange')
        ax.plot(dimensions,p_2_svm,label="svm 2$\sigma$",color='lawngreen')

        ax.plot(dimensions,p_2_miranda_2bins,label="Miranda 2bins 2$\sigma$",color='red')
        ax.plot(dimensions,p_2_miranda_3bins,label="Miranda 3bins 2$\sigma$",color='indianred')
        ax.plot(dimensions,p_2_miranda_4bins,label="Miranda 4bins 2$\sigma$",color='darkorchid')
	ax.plot(dimensions,p_2_miranda_5bins,label="Miranda 5bins 2$\sigma$",color='saddlebrown')
        ax.plot(dimensions,p_2_miranda_6bins,label="Miranda 6bins 2$\sigma$",color='deeppink')
        #ax.plot(dimensions,p_2_miranda_7bins,label="Miranda 7bins 2$\sigma$",color='darkorchid')
        #ax.plot(dimensions,p_2_miranda_8bins,label="Miranda 8bins 2$\sigma$",color='mediumpurple')

        plt.ylim([-5,105])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis sin noCPV")
        ax.legend(loc='right')
        fig_name="sin1diff_2_and_2_periods_noCPV_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_sin/"+fig_name)
        fig.savefig("../svm_sin/"+fig_name)
        fig.savefig("../miranda_sin/"+fig_name)
        print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_optimised':
        dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_sin/"+str(dim)+'Dsin1diff_5_and_6_CPV_optimised_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

        p_1_svm = []
        p_2_svm = []
        p_3_svm = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../svm_sin/"+str(dim)+'Dsin1diff_5_and_6_CPV_optimised_svm_p_values_1_2_3_std_dev.txt')
                p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

        print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_6_periods_p_value_distribution_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_6_periods_p_value_distribution_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_6_periods_p_value_distribution_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
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
        ax.set_title("Dimensionality analysis Sin1Diff")
        ax.legend(loc='right')
        fig_name="sin1diff_5_and_6_periods_optimised_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_sin/"+fig_name)
        fig.savefig("../svm_sin/"+fig_name)
        fig.savefig("../miranda_sin/"+fig_name)
        print("Saved the figure as" , fig_name+".png")

elif MODE == 'ensemble_noCPV_optimised':
        dimensions=[2,3,4,5,6,7,8,9,10]

        p_1_bdt = []
        p_2_bdt = []
        p_3_bdt = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../bdt_sin/"+str(dim)+'Dsin1diff_5_and_5_noCPV_optimised_bdt_p_values_1_2_3_std_dev.txt')
                p_1_bdt.append(temp1), p_2_bdt.append(temp2), p_3_bdt.append(temp3)

        print("Boosted decision tree : ", p_1_bdt,p_2_bdt,p_3_bdt)

        p_1_svm = []
        p_2_svm = []
        p_3_svm = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../svm_sin/"+str(dim)+'Dsin1diff_5_and_5_noCPV_optimised_svm_p_values_1_2_3_std_dev.txt')
                p_1_svm.append(temp1), p_2_svm.append(temp2), p_3_svm.append(temp3)

        print("Support vector machine : ", p_1_svm,p_2_svm,p_3_svm)

        p_1_miranda_2bins = []
        p_2_miranda_2bins = []
        p_3_miranda_2bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_5_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_2_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_2bins.append(temp1), p_2_miranda_2bins.append(temp2), p_3_miranda_2bins.append(temp3)

        print("Miranda 2 bins: ", p_1_miranda_2bins,p_2_miranda_2bins,p_3_miranda_2bins)

        p_1_miranda_3bins = []
        p_2_miranda_3bins = []
        p_3_miranda_3bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_5_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_3_bins_p_values_1_2_3_std_dev.txt')
                p_1_miranda_3bins.append(temp1), p_2_miranda_3bins.append(temp2), p_3_miranda_3bins.append(temp3)

        print("Miranda 3 bins: ", p_1_miranda_3bins,p_2_miranda_3bins,p_3_miranda_3bins)

        p_1_miranda_5bins = []
        p_2_miranda_5bins = []
        p_3_miranda_5bins = []
        for dim in range(2,11):
                temp1,temp2,temp3 = np.loadtxt("../miranda_sin/sin1diff_5_and_5_periods_noCPV_p_value_distribution_miranda_"+str(dim)+'D_5_bins_p_values_1_2_3_std_dev.txt')
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
        ax.set_title("Dimensionality analysis Sin1Diff noCPV")
        ax.legend(loc='right')
        fig_name="sin1diff_5_and_5_periods_noCPV_optimised_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig("../bdt_sin/"+fig_name)
        fig.savefig("../svm_sin/"+fig_name)
        fig.savefig("../miranda_sin/"+fig_name)
        print("Saved the figure as" , fig_name+".png")

else:
	print("No valid mode entered")



