#include <math.h>

#include "nnet.h"
#include <stdio.h>

// Neural Net base
Unit *NeuralNet::GenerateUnitFromType(neuronTypes t, Layer *inputLayer)
{
    switch (t)
    {
    case logistic:
        return new Logistic(inputLayer);
        break;

    case perceptron:
        return new Perceptron(inputLayer);
        break;

    case linear:
        return new Linear(inputLayer);
        break;
    }
    return nullptr;
}

Layer *NeuralNet::operator[](int index)
{
    return this->layers[index];
}

size_t NeuralNet::size()
{
    return this->layers.size();
}

size_t NeuralNet::size(int index)
{
    return (*this)[index]->size();
}

void NeuralNet::Dump()
{
    printf("Neural Net with %lu layers\n", this->layers.size());
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        Layer &l = *(*this)[i];
        printf("Layer %lu (index %lu) - %lu units\n", i, l.Index(), l.size());
        for (size_t j = 0; j < l.size(); j++)
        {
            Unit &u = l[j];
            printf("Neuron %2lu: raw: % .8f, delta % .8f, error % .8f\n\tactivation: %f\n\tweights:", j, u.Raw(), u.delta, u.error, u.Activation());
            auto cw = u.GetConnectionWeights();
            for (auto w : cw)
            {
                printf("% 0.02f, ", w);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("Output value: % f\n", this->Output());
}

void NeuralNet::SetInput(const std::vector<nn_num_t> &values)
{
    // - 1 to account for bias neuron
    if (values.size() != this->layers[0]->size() - 1)
    {
        printf("Error setting input for neural network!\n"
               "Expected %lu input values, received vector of size %lu\n",
               this->layers[0]->size(), values.size());
    }
    InputLayer &il = *static_cast<InputLayer *>(this->layers[0]);
    for (size_t i = 0; i < values.size(); i++)
    {
        // offset by 1 to skip over bias neuron at index 0
        il[i + 1].SetValue(values[i]);
    }
}

const nn_num_t NeuralNet::GetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to)
{
    if ((from.first + 1 > this->size()) ||
        (from.second + 1 > this->size(from.first)) ||
        (from.first + 1 != to.first) ||
        (to.first + 1 > this->size()) ||
        (to.second + 1 > this->size(to.first)))
    {
        printf("Invalid request to get weight from layer %lu:%lu to layer %lu:%lu\n",
               from.first, from.second, to.first, to.second);
        exit(1);
    }
    Layer &tl = *(*this)[to.first];
    return tl[to.second].GetConnectionWeights()[from.second];
}

void NeuralNet::ChangeWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t delta)
{
    if ((from.first + 1 > this->size()) ||
        (from.second + 1 > this->size(from.first)) ||
        (from.first + 1 != to.first) ||
        (to.first + 1 > this->size()) ||
        (to.second + 1 > this->size(to.first)))
    {
        printf("Invalid request to get weight from layer %lu:%lu to layer %lu:%lu\n",
               from.first, from.second, to.first, to.second);
        exit(1);
    }
    Layer &tl = *(*this)[to.first];
    tl[to.second].ChangeConnectionWeight(from.second, delta);
    // return tl[to.second].GetConnectionWeights()[from.second];
}

void NeuralNet::SetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t w)
{
    if ((from.first + 1 > this->size()) ||
        (from.second + 1 > this->size(from.first)) ||
        (from.first + 1 != to.first) ||
        (to.first + 1 > this->size()) ||
        (to.second + 1 > this->size(to.first)))
    {
        printf("Invalid request to get weight from layer %lu:%lu to layer %lu:%lu\n",
               from.first, from.second, to.first, to.second);
        exit(1);
    }
    Layer &tl = *(*this)[to.first];
    tl[to.second].SetConnectionWeight(from.second, w);
    // return tl[to.second].GetConnectionWeights()[from.second];
}

void NeuralNet::AddNeuron(size_t layer, neuronTypes t)
{
    if (layer == 0)
    {
        printf("Use AddInput() to add input (neuron on layer 0)\n");
        exit(1);
    }
    else if (layer == this->size() - 1)
    {
        printf("Can't add output to existing network\n");
        exit(1);
    }
    this->layers[layer]->AddUnit(GenerateUnitFromType(t, this->layers[layer - 1]));
};

void NeuralNet::RemoveNeuron(std::pair<size_t, size_t> index)
{
    if (index.first == 0)
    {
        printf("Can't remove input (neuron from layer 0)\n");
        exit(1);
    }
    else if (index.first == this->size() - 1)
    {
        printf("Can't remove output from existing network\n");
        exit(1);
    }
    else if (index.second == 0)
    {
        printf("Can't remove bias neuron (index 0) from a given layer!\n");
        exit(1);
    }
    else if (this->size(index.first) < index.second + 1)
    {
        printf("Can't remove neuron at index %lu from layer %lu (layer has %lu neurons)\n",
               index.second, index.first, this->size(index.first));
        exit(1);
    }
    this->layers[index.first]->RemoveUnit(index.second);
}

void NeuralNet::AddInput() { this->layers.front()->AddUnit(new Input()); };

