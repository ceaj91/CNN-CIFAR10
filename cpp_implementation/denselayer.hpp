

#include "common.hpp"
using namespace std;

class DenseLayer {
private:
    int input_size;
    int output_size;
    int softmax;
    std::vector<std::vector<float> > dense_weights;
    std::vector<float> dense_bias;
    std::vector<float> output;

public:
    DenseLayer(int input_size, int output_size, int softmax);
    void load_dense_layer(const char* weights_file_name,const char* bias_file_name);
    vector1D forward_prop(std::vector<float> input);
    void GetInfo();
    void GetOutput();
};
