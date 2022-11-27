#include <math.h>

#include "nnet.h"
#include <stdio.h>

void Neuron::Recalculate()
{
    this->value_ = 0.0;
    std::size_t nInputs = this->connectionWeights.size();
    for (std::size_t i = 0; i < nInputs; i++)
    {
        this->value_ += ((*this->inputLayer)[i].Activation() * this->connectionWeights[i]);
    }
}

NeuralNet::NeuralNet(int nInputs)
{
    this->layers.push_back(new Layer(this, true));
    for (int i = 0; i < nInputs; i++)
    {
        this->layers.back()->AddUnit(new Input());
    }
}

void NeuralNet::AddLayer(size_t size, enum neuronTypes t)
{
    OutputLayer *ol = nullptr;
    NeuronLayer *newLayer = nullptr;
    if (this->nOutputs > 0)
    {
        ol = static_cast<OutputLayer *>(this->layers.back());
        this->layers.pop_back();
        newLayer = new NeuronLayer(this, true);

        for (auto u = ol->begin(); u != ol->end(); ++u)
        {
            while ((*u)->GetConnectionWeights().size() < size + 1)
            {
                (*u)->AddConnection(0.01);
            }
            while ((*u)->GetConnectionWeights().size() > size + 1)
            {
                (*u)->RemoveConnection((*u)->GetConnectionWeights().size() - 1);
            }
        }
        ol->SetInputLayer(newLayer);
    }
    else
    {
        newLayer = new NeuronLayer(this, true);
    }
    // newLayer->SetIndex(ol->index());
    for (size_t i = 0; i < size; i++)
    {
        Unit *newU = GenerateUnitFromType(t, this->layers.back());
        newLayer->AddUnit(newU);
    }

    if (ol != nullptr)
    {
        for (size_t i = 0; i < ol->size(); i++)
        {
            for (size_t j = 0; j < this->layers.back()->size(); j++)
            {
                (*newLayer)[i].SetConnectionWeight(j, (*ol)[i].GetConnectionWeights()[j]);
            }
        }
    }
    this->layers.push_back(newLayer);
    if (ol != nullptr)
    {
        ol->SetIndex(this->size());
        this->layers.push_back(ol);
    }
}

void NeuralNet::AddOutputLayer(int size, enum neuronTypes t)
{
    OutputLayer *ol = new OutputLayer(this);
    for (int i = 0; i < size; i++)
    {
        ol->AddUnit(GenerateUnitFromType(t, this->layers.back()));
    }
    this->layers.push_back(ol);
}

void NeuralNet::ConfigureOutput(int nOutputs, enum neuronTypes nt)
{
    this->AddOutputLayer(nOutputs, nt);
    this->nOutputs = nOutputs;
}

nn_num_t NeuralNet::Output()
{
    this->ForwardPropagate();
    switch (this->nOutputs)
    {
    case 1:
        return ((*this->layers.back())[0].Activation());
        break;

    default:
        int maxIndex = 0;
        nn_num_t maxValue = -1.0 * MAXFLOAT;
        OutputLayer &ol = *static_cast<OutputLayer *>(this->layers.back());
        for (size_t i = 0; i < ol.size(); i++)
        {
            nn_num_t thisOutput = ol[i].Activation();
            if (thisOutput > maxValue)
            {
                maxValue = thisOutput;
                maxIndex = i;
            }
        }
        return (nn_num_t)maxIndex;
    }
    return -999.999;
}

void NeuralNet::BackPropagate(const std::vector<nn_num_t> &expectedOutput)
{
    // delta of each output j = activation derivative(j) * (expected(j) - actual(j))
    OutputLayer &ol = *static_cast<OutputLayer *>(this->layers.back());
    for (size_t j = 0; j < ol.size(); j++)
    {
        ol[j].delta = ol[j].ActivationDeriv() * (expectedOutput[j] - ol[j].Activation());
    }

    // for all other layers, delta of a node i in the layer is:
    // activation derivative(i) * sum for all j(weight of connection from i to j * delta(j))
    for (size_t li = this->layers.size() - 2; li > 0; li--)
    {
        Layer &l = *this->layers[li];
        Layer &nl = *this->layers[li + 1];
        for (size_t i = 0; i < l.size(); i++)
        {
            nn_num_t sum = 0.0;
            for (size_t nli = 0; nli < nl.size(); nli++)
            {
                Unit &j = nl[nli];
                sum += (j.GetConnectionWeights()[i] * j.delta);
            }
            l[i].delta = sum * l[i].ActivationDeriv();
        }
    }
}

void NeuralNet::UpdateWeights(nn_num_t learningRate)
{
    for (size_t li = this->size() - 1; li > 0; li--)
    {
        Layer &l = *(*this)[li];
        Layer &pl = *(*this)[li - 1];
        for (auto j = l.begin(); j != l.end(); ++j)
        {
            for (size_t i = 0; i < pl.size(); i++)
            {
                (*j)->ChangeConnectionWeight(i, learningRate * pl[i].Activation() * (*j)->delta);
            }
        }
    }
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

void NeuralNet::ForwardPropagate()
{
    if (this->nOutputs == 0)
    {
        printf("Error - must configure neural net outputs before calling Output() or Learn()\n");
        exit(1);
    }
    for (size_t i = 1; i < this->size(); i++)
    {
        Layer *l = this->operator[](i);
        for (auto u = l->begin(); u != l->end(); ++u)
        {
            (*u)->Recalculate();
        }
    }
}
