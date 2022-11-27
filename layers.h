#ifndef __LAYERS_H_
#define __LAYERS_H_
#include <stdio.h>

#include "units.h"

class NeuralNet;

class Layer
{
    friend class NeuronLayer;
    friend class InputLayer;
    friend class OutputLayer;

private:
    size_t index_;
    std::vector<Unit *> units;
    NeuralNet *myNet;

public:
    Unit &operator[](size_t index);

    Layer(NeuralNet *myNet_, bool addBias);
    ~Layer();

    void AddUnit(Unit *u);
    void RemoveUnit(size_t index);
    
    size_t size();
    size_t Index();

    void SetIndex(size_t i);

    std::vector<Unit *>::iterator begin();
    std::vector<Unit *>::iterator end();
};

class NeuronLayer : public Layer
{
public:
    explicit NeuronLayer(NeuralNet *myNet_, bool addBias);

    Neuron &operator[](int index);

    void SetInputLayer(Layer *l);
};

class InputLayer : public Layer
{
public:
    explicit InputLayer(NeuralNet *myNet_);

    Input &operator[](int index);
};

class OutputLayer : public NeuronLayer
{
public:
    explicit OutputLayer(NeuralNet *myNet_);

    Neuron &operator[](int index);

};

#endif
