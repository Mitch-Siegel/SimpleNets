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

private:
    std::vector<Layer *> layers;
    int nOutputs = 0;
    void AddOutputLayer(int size, enum neuronTypes t);
    Layer *operator[](int index) { return this->layers[index]; };
    NeuronLayer &hiddenLayer(int index) { return *static_cast<NeuronLayer *>((*this)[index]); };
    void BackPropagate(const std::vector<nn_num_t> &expectedOutput);
    void UpdateWeights(nn_num_t learningRate);
    void ForwardPropagate();

    static Unit *GenerateUnitFromType(neuronTypes t, Layer *inputLayer)
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

public:
    explicit NeuralNet(int nInputs);
    ~NeuralNet()
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            delete this->layers[i];
        }
    }
    size_t size() { return this->layers.size(); };
    size_t size(int index) { return (*this)[index]->size(); };

    void AddLayer(size_t size, enum neuronTypes t);

    void ConfigureOutput(int nOutputs, enum neuronTypes nt);

    nn_num_t Output();

    void Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate)
    {
        if(expectedOutput.size() != this->layers.back()->size())
        {
            printf("Provided expected output array of length %lu, expected size %lu\n", 
            expectedOutput.size(), this->layers.back()->size());
            exit(1);
        }
        this->BackPropagate(expectedOutput);
        this->UpdateWeights(learningRate);
    };

    void Dump();

    void SetInput(const std::vector<nn_num_t> &values);

    const nn_num_t GetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to)
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

    void ChangeWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t delta)
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

    void SetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t w)
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

    void AddNeuron(size_t layer, neuronTypes t)
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

    void RemoveNeuron(std::pair<size_t, size_t> index)
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

    void AddInput() { this->layers.front()->AddUnit(new Input()); };
};

#endif
