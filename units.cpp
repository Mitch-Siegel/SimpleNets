#include "units.h"

Unit::~Unit()
{
}

nn_num_t Unit::Raw()
{
    return this->value_;
}
void Unit::ChangeConnectionWeight(int index, nn_num_t delta)
{
    this->connectionWeights[index] += delta;
};

void Unit::SetConnectionWeight(int index, nn_num_t w)
{
    this->connectionWeights[index] = w;
};

nn_num_t Unit::operator[](int index)
{
    return connectionWeights[index];
};

const std::vector<nn_num_t> &Unit::GetConnectionWeights()
{
    return this->connectionWeights;
};

void Unit::AddConnection(nn_num_t weight)
{
    this->connectionWeights.push_back(weight);
};

void Unit::RemoveConnection(size_t index)
{
    this->connectionWeights.erase(this->connectionWeights.begin() + index);
};

// Neuron

Neuron::Neuron(Layer *inputLayer)
{
    this->inputLayer = inputLayer;
}

Neuron::~Neuron()
{
}

void Neuron::Changeconnectionweight(int index, nn_num_t delta)
{
    this->connectionWeights[index] += (this->connectionWeights[index] * delta);
}

void Neuron::SetInputLayer(Layer *newInputLayer)
{
    this->inputLayer = newInputLayer;
}

// Logistic
Logistic::Logistic(Layer *inputLayer) : Neuron(inputLayer)
{
}

nn_num_t Logistic::Activation()
{
    return 1.0 / (1.0 + exp(-1.0 * this->value_));
}

nn_num_t Logistic::ActivationDeriv()
{
    nn_num_t a = this->Activation();
    return a * (1.0 - a);
}

// Perceptron
Perceptron::Perceptron(Layer *inputLayer) : Neuron(inputLayer)
{
}

nn_num_t Perceptron::Activation()
{
    return (this->value_ > 0) ? 1.0 : 0.0;
}

nn_num_t Perceptron::ActivationDeriv()
{
    nn_num_t a = 0.001 / (0.001 + exp(-100.0 * this->value_));
    return a * (1.0 - a);
}

// Linear

Linear::Linear(Layer *inputLayer) : Neuron(inputLayer)
{
}

nn_num_t Linear::Activation()
{
    return this->value_;
}

nn_num_t Linear::ActivationDeriv()
{
    return 1.0;
}

// bias
BiasNeuron::BiasNeuron() : Unit()
{
    this->value_ = 1.0;
}

nn_num_t BiasNeuron::Activation()
{
    return this->value_;
}

nn_num_t BiasNeuron::ActivationDeriv()
{
    return 0.0;
}

void BiasNeuron::Recalculate()
{
}
