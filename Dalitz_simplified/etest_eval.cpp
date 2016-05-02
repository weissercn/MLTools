#include <iostream>
#include <fstream>      // std::ifstream
#include <math.h>       /* exp */
#include <time.h> 
#include <vector>
#include <cstdlib>

using namespace std;



double d_squared(vector<double> first_point, vector<double> second_point, double sigma)
{
	double dx2 =0.0;
	for(int d=0; d < first_point.size(); d++)
        {
                dx2 += pow(second_point[d]-first_point[d],2.0);
        }
	double dx=pow(dx2,0.5);
        return exp(-1*pow(dx,2.0)/(2*pow(sigma,2.0)));

}

int myrandom(int i) { std::srand(std::time(0)); return std::rand() % i; }

int main(int argc, char *argv[])
{
	//string file1_name="/Users/weisser/MIT_Dropbox/MIT/Research/MLTools/Dalitz/dpmodel/data/data_optimisation.0.0.txt";
	//string file2_name="/Users/weisser/MIT_Dropbox/MIT/Research/MLTools/Dalitz/dpmodel/data/data_optimisation.200.1.txt";

	//string file0_name_original="/Users/weisser/MIT_Dropbox/MIT/Research/MLTools/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.0_1.txt";
	//string file1_name_original="/Users/weisser/MIT_Dropbox/MIT/Research/MLTools/Dalitz/gaussian_samples/higher_dimensional_gauss/gauss_data/data_high4Dgauss_optimisation_10000_0.5_0.1_0.01_1.txt";
	//double sigma = 0.2;

	string file0_name_original= argv[1];
	string file1_name_original= argv[2];
	double sigma = atof(argv[3]);
	int permutation_mode = atoi(argv[4]);
	
	double T;
	vector< vector<string> > comp_file_list;
	
	vector<string> file_name_temp;
	file_name_temp.push_back(file0_name_original);
	file_name_temp.push_back(file1_name_original);

	comp_file_list.push_back(file_name_temp);

	int file_counter;
	for(file_counter=0; file_counter < comp_file_list.size(); file_counter++)
	{
		string file0_name = comp_file_list[file_counter][0];
		string file1_name = comp_file_list[file_counter][1];
		cout << "Operating on files " << file0_name << " and " << file1_name <<" permutation_mode : " << permutation_mode <<endl;
		
		vector< vector<double> > features_0, features_1;

		ifstream file0 (file0_name);

		while(!file0.eof())
		{
			double a, b;
			file0 >> a >> b; // extracts 2 floating point values seperated by whitespace
			vector <double> temp;
			temp.push_back(a);
			temp.push_back(b);
			features_0.push_back(temp);	
		}

		cout << endl << "features_0 : " <<endl;
		for(int j=0; j < 10; j++){cout << features_0[j][0] << " " << features_0[j][1] <<endl;}

		ifstream file1 (file1_name);

		while(!file1.eof())
		{   
			double a, b;
			file1 >> a >> b; // extracts 2 floating point values seperated by whitespace
			vector <double> temp;
			temp.push_back(a);
			temp.push_back(b);
			features_1.push_back(temp);    
		}   

		cout << endl << "features_1 : " <<endl;
		for(int j=0; j < 10; j++){cout << features_1[j][0] << " " << features_1[j][1]  <<endl;}

		if (permutation_mode !=0)
		{
			cout << "Permutation mode. The data will be shuffled." <<endl;
			vector< vector<double> > features_01 = features_0;
			features_01.insert(features_01.end(), features_1.begin(), features_1.end());
			random_shuffle ( features_01.begin(), features_01.end(),myrandom );
			size_t const half_size = features_0.size(); 
			vector< vector<double> > split_lo(features_01.begin(), features_01.begin() + half_size);
			vector< vector<double> >split_hi(features_01.begin() + half_size, features_01.end());
			features_0 = split_lo;
			features_1 = split_hi;
			cout << "Done with the data shuffling" <<endl;
			cout << endl << "features_0 : " <<endl;
			for(int j=0; j < 10; j++){cout << features_0[j][0] << " " << features_0[j][1] <<endl;}
			cout << endl << "features_1 : " <<endl;
			for(int j=0; j < 10; j++){cout << features_1[j][0] << " " << features_1[j][1] <<endl;}
		}

		double no_0=features_0.size();
		double no_1=features_1.size();
		double no_2=no_0+no_1;
		cout << "no_0 : " << no_0 <<" no_1 : " << no_1 << endl;	
		time_t beginning_of_loop;
		time_t now;
		time(&beginning_of_loop);

		double T_1st_contrib=0.0;
                cout << "Calculating the first contribution to T"<<endl;
		for(int i=0; i < no_0; i++)
		{
                        if(i % 1000==0){time(&now);cout <<"Point : " << i << " Time since beginning of loop in seconds : " << difftime(now,beginning_of_loop) <<endl;}
			for(int j=i+1; j < no_0; j++)
                                T_1st_contrib += d_squared(features_0[i],features_0[j],sigma);
                }
		T_1st_contrib = T_1st_contrib/(no_0*(no_0-1.0));

                double T_2nd_contrib=0;
                cout << "Calculating the second contribution to T"<<endl;
                for(int i=0; i < no_1; i++)
		{		
                        if(i % 1000==0){time(&now);cout <<"Point : " << i << " Time since beginning of loop in seconds : " << difftime(now,beginning_of_loop) <<endl;}
			for(int j=i+1; j < no_1; j++)
			{        
				T_2nd_contrib += d_squared(features_1[i],features_1[j],sigma);
                	}
		}
		T_2nd_contrib = T_2nd_contrib/(no_1*(no_1-1));


                double T_3rd_contrib=0;
                cout << "Calculating the third contribution to T"<<endl;
                for(int i=0; i < no_1; i++)
                { 
                        if(i % 1000==0){time(&now);cout <<"Point : " << i << " Time since beginning of loop in seconds : " << difftime(now,beginning_of_loop) <<endl;}
			for(int j=0; j < no_1; j++)
                        {         
				T_3rd_contrib += d_squared(features_0[i],features_1[j],sigma);
                	}
		}
		T_3rd_contrib = T_3rd_contrib/(no_2*(no_2-1));

                T = T_1st_contrib + T_2nd_contrib +  T_3rd_contrib;
		cout<<T<<endl;
	}

	std::ofstream outfile;

  	outfile.open("etest_T_values_validation_file.txt", std::ios_base::app);
  	outfile << T << endl; 
	cout << "Finished writing to file " <<endl;

  	return 0;


}
