#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <functional>
#include <iomanip>

#include "denselayer.hpp"

DenseLayer::DenseLayer(int input_size, int output_size, int softmax) {
        this->input_size = input_size;
        this->output_size = output_size;
        this->softmax = softmax;

        // Postavljamo velicinu vectora weightsa da bude ista kao i input_size
        // Tako sto ga prosirujemo kontenerima koji sadrze 0
        dense_weights.resize(input_size, std::vector<float>(output_size, 0));

        // Postavljamo velicinu vectora biasa da bude ista kao i output_size
        dense_bias.resize(output_size, 0);
    }


void DenseLayer::load_dense_layer(const char* weights_file_name,const char* bias_file_name) {
    std::ifstream weights_file(weights_file_name, std::ifstream::in);
    std::ifstream bias_file(bias_file_name, std::ifstream::in);

    for (int i = 0; i < input_size; ++i) {
        std::string line;
        std::getline(weights_file, line);
        std::istringstream iss(line);
        std::vector<float> float_values;
        for (int j = 0; j < output_size; ++j) {
              float value;
              iss >> value;
              float_values.push_back(value);
        }
        dense_weights[i] = float_values;
    }

    for (int i = 0; i < output_size; ++i) {
        std::string line;
        std::getline(bias_file, line);
        float value = std::stof(line); // string to float
        dense_bias[i] = value;
    }

    weights_file.close();
    bias_file.close();
}



vector1D DenseLayer::forward_prop(std::vector<float> input) {

    // Napravi vector temp koji je size = output_size
    // Inicijalizuj 0 u svakom kontejneru
    std::vector<float> temp(output_size, 0);

    // Za svaku OUTPUT NODU mnozi svaku INPUT NODU sa odgovarajucim WEIGHTOM
    // I na kraju dodaje bias za svaku OUTPUT NODU
    // Ekvivalent: temp = np.dot(input[0:],self.dense_weights) + self.dense_bias
    for (int i = 0; i < output_size; i++) {
            for (int k = 0; k < input_size; k++) {
                temp[i] += input[k] * dense_weights[k][i];
            }
        temp[i] += dense_bias[i];
    }

    if (!softmax) {
            for (int i = 0; i < output_size; i++) {
                        // Sve negativne koef normalizuje na 0
                        // Ekvivalent: output = np.maximum(0,temp)
                    if(temp[i] < 0) temp[i] = 0;
            }
    } else {
            // Pronalazi najveci element u temp vectoru
        float max_val = *max_element(temp.begin(), temp.end());
        float sum_exp = 0.0;
        for (int j = 0; j < output_size; j++) {
                // Uredjuje temp vector tako sto od starih vrednosti njegovih kontejnera
                // Oduzima vrednosti max elementa, i eksponira na e (skalira)
                // Ova for petlja je ekvivalent:
                // output = temp - np.max(temp,axis=1,keepdims=True)
                // exp_out = np.exp(output)
                // sum_exp_out = np.sum(exp_out,axis=1,keepdims=True)
            temp[j] = exp(temp[j] - max_val);
            sum_exp += temp[j];
        }
        for (int j = 0; j < output_size; j++) {
                // Ekvivalent: output = exp_out / sum_exp_out
            temp[j] /= sum_exp;
        }
    }
    output = temp;
    return output;
}


void DenseLayer::GetInfo() {
	std::cout << "Input size is " << input_size << endl;
	std::cout << "Output size is " << output_size << endl;
	std::cout << "Softmax ? " << softmax << endl;
	std::cout << setprecision(20);
	std::cout << "First weight is: " << dense_weights[0][0] << endl;
	std::cout << "First bias is: " << dense_bias[0] << endl;
}


void DenseLayer::GetOutput() {
	std::cout << "------------------" << endl;
	std::cout << "Output is " << endl;
	for(auto i = output.begin(); i != output.end(); ++i)
		cout << *i << endl;
	std::cout << "------------------" << endl;
}
