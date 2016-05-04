from __future__ import print_function 
import os
import sys

comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.0.0.txt",os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.200.1.txt")]
#comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.0_1.txt",os.environ['MLToolsDir']+"/Dalitz/ga    ussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.01_1.txt")] 

sigma = 0.2

for counter in range(100):
	command = ( os.environ['MLToolsDir'] + "/Dalitz_simplified/etest_eval.out " +comp_file_list[0][0] + " " + comp_file_list[0][1] + " " + str(sigma) + " " + str(1) )
	print(command)
	os.system(command)




 
