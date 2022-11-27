#ifndef __UNITS_H_
#define __UNITS_H_

#include <vector>
#include <math.h>

typedef float nn_num_t;

class Layer;

/*
 * A unit represents a node in the network
 * connections flow from left to right (lower to higher layer indices)
 */
class Unit
{
    friend class Layer;

protected:
    std::vector<nn_num_t> connectionWeights;
    nn_num_t value_ = 0.0;

public:
    nn_num_t delta = 0.0;
    nn_num_t error = 0.0;
    virtual ~Unit() = 0;
    nn_num_t Raw();
    virtual nn_num_t Activation() = 0;
    virtual nn_num_t ActivationDeriv() = 0;
    virtual void Recalculate() = 0;
    void ChangeConnectionWeight(int index, nn_num_t delta);
    void SetConnectionWeight(int index, nn_num_t w);
    nn_num_t operator[](int index);
    const std::vector<nn_num_t> &GetConnectionWeights();
    void AddConnection(nn_num_t weight);
    void RemoveConnection(size_t index);
};

class Input : public Unit
{
public:
    explicit Input() { value_ = 0.0; };
    ~Input(){};

    nn_num_t Activation() override { return value_; };
    nn_num_t ActivationDeriv() override { return 0.0; };
    void Recalculate() override{};
    void SetValue(nn_num_t newValue) { this->value_ = newValue; };
    void Backpropagate(nn_num_t error, nn_num_t alpha){};
};

class Neuron : public Unit
{
    friend class Layer;
    friend class NeuronLayer;
    friend class OutputLayer;

private:
    Layer *inputLayer;
    void Changeconnectionweight(int index, nn_num_t delta);
    void SetInputLayer(Layer *newInputLayer);

public:
    Neuron(Layer *inputLayer);
    ~Neuron();

    void Recalculate() override;
    void Backpropagate(nn_num_t error, nn_num_t alpha);
};

class Logistic : public Neuron
{
public:
    explicit Logistic(Layer *inputLayer);
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
};

class Perceptron : public Neuron
{
public:
    Perceptron(Layer *inputLayer);
    nn_num_t Activation();
    nn_num_t ActivationDeriv() override;
};

class Linear : public Neuron
{
public:
    explicit Linear(Layer *inputLayer);
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
};

class BiasNeuron : public Unit
{
public:
    explicit BiasNeuron();
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
    void Recalculate() override;
};

#endif
