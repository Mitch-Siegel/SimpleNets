#ifndef __UNITS_H_
#define __UNITS_H_

// #include <vector>
#include <set>
#include <math.h>
#include <stdint.h>

typedef float nn_num_t;

enum neuronTypes
{
    input,
    bias,
    logistic,
    perceptron,
    linear,
};

const char *GetNeuronTypeName(neuronTypes t);

class Layer;
class Unit;
// const size_t Unit::Id();

class Connection
{
private:
    bool idOnly = false;

public:
    union
    {
        Unit *u;
        size_t id;
    } from;

    nn_num_t weight;

    Connection(Unit *u, nn_num_t weight);
    explicit Connection(Unit *u);
    explicit Connection(size_t id);

    bool operator==(const Connection &b);

    bool operator<(const Connection &b) const;
};

/*
 * A unit represents a node in the network
 * connections flow from left to right (lower to higher layer indices)
 */
class Unit
{
    friend class Layer;

private:
    size_t id_;
    neuronTypes type_;

protected:
    // std::vector<nn_num_t> connectionWeights;
    std::set<Connection> connections;
    nn_num_t value_ = 0.0;

public:
    nn_num_t delta = 0.0;
    nn_num_t error = 0.0;
    Unit(size_t id, neuronTypes type);
    virtual ~Unit() = 0;
    const size_t Id();
    const neuronTypes type();
    nn_num_t Raw();
    virtual nn_num_t Activation() = 0;
    virtual nn_num_t ActivationDeriv() = 0;
    virtual void CalculateValue() = 0;
    void ChangeConnectionWeight(Unit *from, nn_num_t delta);
    void SetConnectionWeight(Unit *from, nn_num_t w);

    const std::set<Connection> &GetConnections();
    void AddConnection(Unit *u, nn_num_t w);
    void RemoveConnection(Unit *u);
    void Disconnect();
};

class Input : public Unit
{
public:
    explicit Input(size_t id);
    ~Input(){};

    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
    void CalculateValue() override;
    void SetValue(nn_num_t newValue);
};

class Neuron : public Unit
{
    friend class Layer;
    friend class NeuronLayer;
    friend class OutputLayer;

private:
    void Changeconnectionweight(int index, nn_num_t delta);

public:
    explicit Neuron(size_t id, neuronTypes type);
    ~Neuron();

    void CalculateValue() override;
    void Backpropagate(nn_num_t error, nn_num_t alpha);
};

class Logistic : public Neuron
{
public:
    explicit Logistic(size_t id);
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
};

class Perceptron : public Neuron
{
public:
    explicit Perceptron(size_t id);
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
};

class Linear : public Neuron
{
public:
    explicit Linear(size_t id);
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
};

class BiasNeuron : public Unit
{
public:
    explicit BiasNeuron(size_t id);
    nn_num_t Activation() override;
    nn_num_t ActivationDeriv() override;
    void CalculateValue() override;
};

#endif
