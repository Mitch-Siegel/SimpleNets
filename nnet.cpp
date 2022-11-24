#include <math.h>

#include "nnet.h"
#include <stdio.h>

void Neuron::recalculate()
{
    this->value_ = 0.0;
    std::size_t nInputs = this->connectionWeights.size();
    for (std::size_t i = 0; i < nInputs; i++)
    {
        this->value_ += ((*this->inputLayer)[i].activation() * this->connectionWeights[i]);
    }
}

Layer::Layer(NeuralNet *myNet_, bool addBias)
{
    this->myNet = myNet_;
    this->index_ = myNet_->size();

    if (addBias)
    {
        this->AddUnit(new BiasNeuron());
    }
}

void Layer::AddUnit(Unit *u)
{
    this->units.push_back(u);
    if (this->index_ > 0)
    {
        for (size_t i = 0; i < this->myNet->operator[](this->index_ - 1)->size(); i++)
        {
            u->addConnection(0.01);
        }
    }

    if (this->index_ + 1 < this->myNet->size())
    {
        NeuronLayer &forwardLayer = this->myNet->hiddenLayer(this->index_ + 1);
        std::size_t forwardLayerSize = forwardLayer.size();
        for (std::size_t i = 0; i < forwardLayerSize; i++)
        {
            forwardLayer[i].addConnection(0.01);
        }
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

void NeuralNet::AddLayer(int size, enum neuronTypes t)
{
    NeuronLayer *newLayer = new NeuronLayer(this);
    for (int i = 0; i < size; i++)
    {
        switch (t)
        {
        case logistic:
            newLayer->AddUnit(new Logistic(this->layers.back()));
            break;

        case perceptron:
            newLayer->AddUnit(new Perceptron(this->layers.back()));
            break;

        case linear:
            newLayer->AddUnit(new Linear(this->layers.back()));
            break;
        }
    }
    this->layers.push_back(newLayer);
}

void NeuralNet::AddOutputLayer(int size, enum neuronTypes t)
{
    OutputLayer *ol = new OutputLayer(this);
    for (int i = 0; i < size; i++)
    {
        switch (t)
        {
        case logistic:
            ol->AddUnit(new Logistic(this->layers.back()));
            break;

        case perceptron:
            ol->AddUnit(new Perceptron(this->layers.back()));
            break;

        case linear:
            ol->AddUnit(new Linear(this->layers.back()));
            break;
        }
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
    switch (this->nOutputs)
    {
    case 1:
        return ((*this->layers.back())[0].activation());
        break;

    default:
        int maxIndex = 0;
        nn_num_t maxValue = -1.0 * MAXFLOAT;
        OutputLayer &ol = *static_cast<OutputLayer *>(this->layers.back());
        for (size_t i = 0; i < ol.size(); i++)
        {
            nn_num_t thisOutput = ol[i].activation();
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

void NeuralNet::BackPropagate(std::vector<nn_num_t> expectedOutput)
{
    /*
    // for each node j in the output layer do
    //     Delta[j] <- g'(in_j) \times (y_j - a_j)
    // for l = L-1 to 1 do
    //     for each node i in layer l do
    //         Delta[i] <- g'(in_i) * \sum_j w_ij Delta[j]
    // for each weight w_ij in network do
    //     w_ij <- w_ij + alpha * a_i * delta_j
    */
    // delta of each output j = activation derivative(j) * (expected(j) - actual(j))
    OutputLayer &ol = *static_cast<OutputLayer *>(this->layers.back());
    for(size_t j = 0; j < ol.size(); j++)
    {
        ol[j].delta = ol[j].activationDeriv() * (expectedOutput[j] - ol[j].activation());
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
            l[i].delta = sum * l[i].activationDeriv();
        }
    }

}

void NeuralNet::UpdateWeights(nn_num_t learningRate)
{
    printf("updating weights with learning rate %f\n", learningRate);
    for (size_t li = this->size() - 1; li > 0; li--)
    {
        Layer &l = *(*this)[li];
        Layer &pl = *(*this)[li - 1];
        for (auto j = l.begin(); j != l.end(); ++j)
        {
            for (size_t i = 0; i < pl.size(); i++)
            {
                (*j)->changeconnectionweight(i, learningRate * pl[i].activation() * (*j)->delta);
            }
        }
    }

}

void NeuralNet::dump()
{
    printf("Neural Net with %ld layers\n", this->layers.size());
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        Layer &l = *(*this)[i];
        printf("Layer %ld - %ld units\n", i, l.size());
        for (size_t j = 0; j < l.size(); j++)
        {
            Unit &u = l[j];
            printf("Neuron %2lu: raw: % .8f, delta % .8f, error % .8f\n\tactivation: %f\n\tweights:", j, u.raw(), u.delta, u.error, u.activation());
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

void NeuralNet::setInput(std::vector<nn_num_t> values)
{
    // - 1 to account for bias neuron
    if (values.size() != this->layers[0]->size() - 1)
    {
        printf("Error setting input for neural network!\n"
        "Expected %lu input values, received vector of size %lu\n", this->layers[0]->size(), values.size());
    }
    InputLayer &il = *static_cast<InputLayer *>(this->layers[0]);
    for(size_t i = 0; i < values.size(); i++)
    {
        // offset by 1 to skip over bias neuron at index 0
        il[i + 1].setValue(values[i]);
    }
}

void NeuralNet::ForwardPropagate()
{
    for (size_t i = 1; i < this->size(); i++)
    {
        Layer *l = this->operator[](i);
        for (auto u = l->begin(); u != l->end(); ++u)
        {
            (*u)->recalculate();
        }
    }
}
