#ifndef MAXPOOLLAYER_HPP_INCLUDED
#define MAXPOOLLAYER_HPP_INCLUDED
#include "common.hpp"
class MaxPoolLayer {
private:
    std::vector<int> pool_size;
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    std::vector<std::vector<std::vector<std::vector<float> > > > output;
public:
    MaxPoolLayer(int param_pool_size);
    vector4D forward_prop(std::vector<std::vector<std::vector<std::vector<float> > > > input, std::vector<int> stride);
    void GetInfo();
};

#endif // MAXPOOLLAYER_HPP_INCLUDED
