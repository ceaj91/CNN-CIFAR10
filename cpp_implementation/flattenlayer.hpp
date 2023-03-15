#ifndef FLATTENLAYER_HPP_INCLUDED
#define FLATTENLAYER_HPP_INCLUDED

class FlattenLayer {
private:
    std::vector<std::vector<float>> output;
public:
    std::vector<std::vector<float>> forward_prop(std::vector<std::vector<std::vector<std::vector<float>>>> input);
};

#endif // FLATTENLAYER_HPP_INCLUDED
