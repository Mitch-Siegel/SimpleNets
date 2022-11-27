#include "feedforwardnn.h"

FeedForwardNeuralNet::FeedForwardNeuralNet(int nInputs)
{
    this->layers.push_back(new Layer(this, true));
    for (int i = 0; i < nInputs; i++)
    {
        this->layers.back()->AddUnit(new Input());
    }
}

FeedForwardNeuralNet::~FeedForwardNeuralNet()
{
    for (size_t i = 0; i < this->size(); i++)
    {
        delete this->layers[i];
    }
}

void FeedForwardNeuralNet::AddLayer(size_t size, enum neuronTypes t)
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

void FeedForwardNeuralNet::AddOutputLayer(int size, enum neuronTypes t)
{
    OutputLayer *ol = new OutputLayer(this);
    for (int i = 0; i < size; i++)
    {
        ol->AddUnit(GenerateUnitFromType(t, this->layers.back()));
    }
    this->layers.push_back(ol);
}

void FeedForwardNeuralNet::ConfigureOutput(int nOutputs, enum neuronTypes nt)
{
    this->AddOutputLayer(nOutputs, nt);
    this->nOutputs = nOutputs;
}

void FeedForwardNeuralNet::Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate)

{
    if (expectedOutput.size() != this->layers.back()->size())
    {
        printf("Provided expected output array of length %lu, expected size %lu\n",
               expectedOutput.size(), this->layers.back()->size());
        exit(1);
    }
    this->BackPropagate(expectedOutput);
    this->UpdateWeights(learningRate);
}

nn_num_t FeedForwardNeuralNet::Output()
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

void FeedForwardNeuralNet::BackPropagate(const std::vector<nn_num_t> &expectedOutput)
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

void FeedForwardNeuralNet::UpdateWeights(nn_num_t learningRate)
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

void FeedForwardNeuralNet::ForwardPropagate()
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