#ifndef __NNET_H_
#define __NNET_H_

#include <math.h>
#include <vector>

#include "layers.h"

class Unit;
class Layer;
class InputLayer;
class NeuronLayer;
class OutputLayer;

class NeuralNet
{
    friend class Layer;

public:
    enum neuronTypes
    {
        logistic,
        perceptron,
        linear,
    };

protected:
    static Unit *GenerateUnitFromType(neuronTypes t, Layer *inputLayer);

    std::vector<Layer *> layers;

public:
    Layer *operator[](int index);
    size_t size();
    size_t size(int index);
    virtual nn_num_t Output() = 0;

    void Dump();

    void SetInput(const std::vector<nn_num_t> &values);

    const nn_num_t GetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to);
    void ChangeWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t delta);
    void SetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t w);

    void AddNeuron(size_t layer, neuronTypes t);
    void RemoveNeuron(std::pair<size_t, size_t> index);

    void AddInput();
};


#endif
