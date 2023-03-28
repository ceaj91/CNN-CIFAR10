#include "ConvLayer.hpp"
#include <iostream>
#include <systemc>
#include <fstream>
#include <string>
#include <cstdio>
#include "common.hpp"
#include <iomanip>
#include "../cpp_implementation/MaxPoolLayer.hpp"
#include "../cpp_implementation/flattenlayer.hpp"
#include "../cpp_implementation/denselayer.hpp"
int sc_main(int argc, char* argv[]) {
	//UCITAVANJE SLIKE NE RADI, DOTLE SI STIGAO  "slika.txt"
	//kada ucitas sliku, implementiraj forward_propagation
	vector4D output;
	vector4D output2;
	vector4D output3;
	vector4D output4;
	vector4D output5;
	vector4D output6;
	vector2D output7;
	vector2D output8;
	vector2D output9;

	vector4D slika(1,vector3D(32,vector2D(32,vector1D(3,0.0))));
	ifstream file_slike;
	ifstream file_labele;
	ofstream file_info;
	double num;
	int labela;
	int index=0;
	float max=0;
	double broj_pogodaka=0.0;
	double min_tacnost=1.1;
	double tacnost;
	int W;
	int F;
	const char *weights1 = "../parametars/conv1/conv1_filters.txt";
	const char *bias1 = "../parametars/conv1/conv1_bias.txt";
	const char *weights2 = "../parametars/conv2/conv2_filters.txt";
	const char *bias2 = "../parametars/conv2/conv2_bias.txt";
	const char *weights3 = "../parametars/conv3/conv3_filters.txt";
	const char *bias3 = "../parametars/conv3/conv3_bias.txt";
	const char *dense1_weights = "../parametars/dense1/dense1_weights.txt";
	const char *dense1_bias = "../parametars/dense1/dense1_bias.txt";
	const char *dense2_weights = "../parametars/dense2/dense2_weights.txt";
	const char *dense2_bias = "../parametars/dense2/dense2_bias.txt";

	W=18;
	F=4;

	for(int s = 14; s<=19;s++)
	{
		for(int b=3;b<=5;b++)
		{
			W=s;
			F=b;
			broj_pogodaka=0;
			tacnost=0;
			ConvLayer conv1(3,3,32,"same",W,F);
			MaxPoolLayer maxpool1(2);
			ConvLayer conv2(3,32,32,"same",W,F);
			MaxPoolLayer maxpool2(2);
			ConvLayer conv3(3,32,64,"same",W,F);
			MaxPoolLayer maxpool3(2);
			FlattenLayer flatten1;
			DenseLayer dense1(1024,512,0);
			DenseLayer dense2(512,10,1);

			conv1.load_weights(weights1,bias1);
			conv2.load_weights(weights2,bias2);
			conv3.load_weights(weights3,bias3);
			dense1.load_dense_layer(dense1_weights,dense1_bias);
			dense2.load_dense_layer(dense2_weights,dense2_bias);
			//conv1.print_weights();
			cout<<"Format: "<<F<<"."<<W-F<<endl;
			file_slike.open("../slike/slike.txt");
			file_labele.open("../slike/labele.txt");
			for (int pic_num = 0; pic_num < 10; pic_num++)
			{
				
				file_labele >> labela;
				//cout <<"labela:"<<labela<<" ";
				for (int channel = 0; channel < INPUT_CHANNEL_SIZE; channel++)
				{
				
					for (int row = 0; row < INPUT_PICTURE_SIZE; row++)
					{
						for (int column = 0; column < INPUT_PICTURE_SIZE; column++)
						{
							file_slike >> num;
						
							slika[0][row][column][channel] = num/255.0;
						//std::cout<<slika[0][row][column][channel] <<" ";
						}
						//std::cout<<std::endl;
					}
				}

				output = conv1.forward_prop(slika);
				/*for (int i = 0; i < 32; ++i)
				{
					cout<<output[0][0][0][i]<<endl;
				}*/
				output2 = maxpool1.forward_prop(output,{});
				output3 = conv2.forward_prop(output2);
				output4 = maxpool2.forward_prop(output3,{});
				output5 = conv3.forward_prop(output4);
				output6 = maxpool3.forward_prop(output5,{});
				output7 = flatten1.forward_prop(output6);
				output8 = dense1.forward_prop(output7);
				output9 = dense2.forward_prop(output8);
				max=output9[0][0];
				index=0;
				for (int i = 1; i < 10; i++)
				{
					if(output9[0][i] > max){
						max = output9[0][i];
						index=i;
					}
				}
				//cout<<"index: "<<index<<" ";
				if(labela == index){
					
					broj_pogodaka++;
					//std::cout<<pic_num<<". POGODAK!"<<std::endl;
					
				}
				//else
				//	std::cout<<pic_num<<". PROMASAJ!"<<std::endl;
			/*
				if((pic_num+1) % 100 == 0)
				{
					
					std::cout<<"Tacnost mreze posle "<<pic_num+1<<" slika: "<<broj_pogodaka/((pic_num+1)*1.0)*100<<"%"<<std::endl;
					
				}
				*/
			}

			//tacnost = broj_pogodaka/10000.0 * 100;
			//std::cout<<"Tacnost mreze ukupno: "<<tacnost<<std::endl;
			file_info.open("bit_analiza.txt",ios::app);
			file_info<<"Format: "<<F<<"."<<W-F<<" ("<<s<<" bita)"<<" daje tacnost : "<<tacnost<<"%"<<std::endl;
			std::cout<<"Format: "<<F<<"."<<W-F<<" ("<<s<<" bita)"<<" daje tacnost : "<<tacnost<<"%"<<std::endl;
			file_info.close();
			file_slike.close();
			file_labele.close();
			
		}
	}




	
	
	return 0;
}
