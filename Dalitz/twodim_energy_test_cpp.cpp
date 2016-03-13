////////////////////////////////////
//   twodim_energy_test_cpp.cpp    //
//      Constantin Weisser        //
//	 weisser@mit.edu          //
////////////////////////////////////

#include <iostream>
#include <stdio.h>      /* printf */
#include <math.h>       /* exp */

using namespace std;

double weighting_function(double dx,double sigma)
{
	return exp(-pow(dx,2.0)/(2*pow(sigma,2.0)));	

}

int main() {
	std::cout << "Hello World!" << std::endl;
	std::cin.get();
	return 0;
}




