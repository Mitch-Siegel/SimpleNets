#include <math.h>

#include "nnet.h"
#include <stdio.h>

Neuron::Neuron()
{
    this->inputLayer = nullptr;
    this->delta = 0.0;
}

Neuron::Neuron(Layer *inputLayer)
{
    this->inputLayer = inputLayer;
    this->delta = 0.0;
}

nn_num_t Neuron::output()
{
    if (needRecompute)
    {
        std::size_t nInputs = this->connectionWeights.size();
        this->value = 0;
        for (std::size_t i = 0; i < nInputs; i++)
        {
            this->value += this->inputLayer->operator[](i)->output() * this->connectionWeights[i];
        }
        needRecompute = false;
    }
    return activation(this->value);
}

Perceptron::Perceptron(Layer *inputLayer) : Neuron(inputLayer)
{
}

Perceptron::Perceptron() : Neuron()
{
}

nn_num_t Perceptron::activation(nn_num_t input)
{
    return 1.0 / (1.0 + exp(-1 * input));
}

Layer::Layer(NeuralNet *myNet_, bool addBias)
{
    this->index_ = myNet_->size();

    if (addBias)
    {
        this->units.push_back(new BiasNeuron());
    }

    this->myNet = myNet_;
}

void Layer::AddUnit(Unit *u)
{
    this->units.push_back(u);
    if (this->index_ < this->myNet->size() - 1)
    {
        NeuronLayer *forwardLayer = static_cast<NeuronLayer *>(this->myNet->operator[](this->index_ + 1));
        std::size_t forwardLayerSize = forwardLayer->size();
        for (std::size_t i = 0; i < forwardLayerSize; i++)
        {
            forwardLayer->operator[](i)->connectionWeights.push_back(0);
        }
    }
}

void Neuron::backpropagate(nn_num_t error, nn_num_t alpha)
{
    // for each node j in the output layer do
    //     Delta[j] <- g'(in_j) \times (y_j - a_j)
    // for l = L-1 to 1 do
    //     for each node i in layer l do
    //         Delta[i] <- g'(in_i) * \sum_j w_ij Delta[j]
    // for each weight w_ij in network do
    //     w_ij <- w_ij + alpha * a_i * delta_j

    /*
    function BACK-PROP-LEARNING(examples, network ) returns a neural network
    inputs: examples, a set of examples, each with input vector x and output vector y
    network , a multilayer network with L layers, weights wi,j , activation function g
    local variables: Δ, a vector of errors, indexed by network node
    repeat
    for each weight wi,j in network do
    wi,j ←a small random number
    for each example (x, y) in examples do
    //Propagate the inputs forward to compute the outputs
    for each node i in the input layer do
    ai ←xi
    for  = 2 to L do
    for each node j in layer  do
    inj ←
    i wi,j ai
    aj ← g(inj )
    // Propagate deltas backward from output layer to input layer
    for each node j in the output layer do
    Δ[j] ← g
    (inj ) × (yj − aj )

    for  = L − 1 to 1 do
    for each node i in layer  do
    Δ[i] ← g
    (ini)

    j wi,j Δ[j]

    // Update every weight in network using deltas
    for each weight wi,j in network do
    wi,j ← wi,j + α × ai × Δ[j]
    until some stopping criterion*/

    if (this->inputLayer == nullptr)
    {
        return;
    }

    this->delta = this->output() *

    /*size_t nUnits = this->inputLayer->size();
    for (size_t i = 0; i < nUnits; i++)
    {
        this->inputLayer->operator[](i)->changeconnectionweight(i, (error * alpha));
    }
    if (this->inputLayer->index() > 0)
    {
        size_t inputLayerSize = this->inputLayer->size();
        for (size_t i = 0; i < inputLayerSize; i++)
        {
            // this->myNet->opera

            this->inputLayer->operator[](i)->backpropagate(error * this->connectionWeights[i], alpha);
        }
    }*/
}

void Layer::StartBackProp()
{
    size_t thisSize = this->size();
    for (size_t i = 0; i < thisSize; i++)
    {
        this->operator[](i)->delta = 0;
    }
}

NeuralNet::NeuralNet(int nInputs)
{
    this->layers.push_back(new Layer(this, true));
    for (int i = 0; i < nInputs; i++)
    {
        this->layers.back()->AddUnit(new Input());
    }
    // this->layers.push_back(Layer(this));
}

void NeuralNet::AddLayer(int size, enum neuronTypes t)
{
    this->layers.push_back(new NeuronLayer(this));
    for (int i = 0; i < size; i++)
    {
        switch (t)
        {
        case perceptron:
            this->layers.back()->AddUnit(new Perceptron());
            break;
        }
    }
}

void NeuralNet::AddOutputLayer(int size, enum neuronTypes t)
{
    this->layers.push_back(new OutputLayer(this));
    for (int i = 0; i < size; i++)
    {
        switch (t)
        {
        case perceptron:
            this->layers.back()->AddUnit(new Perceptron());
            break;
        }
    }
}

void NeuralNet::ConfigureOutput(enum outputTypes t, enum neuronTypes nt)
{
    switch (t)
    {
    case output_null:
        printf("Can't set output type to null!\n");
        exit(1);

    default:
        this->AddOutputLayer(1, nt);
        this->outputType = t;
        break;
    }
}

nn_num_t NeuralNet::Output()
{
    switch (this->outputType)
    {
    case output_singular:
        return (this->layers.back()->operator[](0)->output());
        break;

    case output_null:
        printf("Request from output from an unconfigured neural net!\nMust call ConfigreOutput() during setup!\n");
        exit(1);
    }
    return -999.999;
}

void NeuralNet::BackPropagate(nn_num_t expectedOutput, nn_num_t alpha)
{
    for (size_t i = 1; i < this->layers.size(); i++)
    {
        this->layers[i]->StartBackProp();
    }
    switch (output_singular)
    {
    case output_singular:
        this->layers.back()->BackPropagate(this->Output() - expectedOutput, alpha);
        break;

    case output_null:
        printf("Call to BackPropagate() from an unconfigured neural net!\nMust call ConfigreOutput() during setup!\n");
        exit(1);
        break;
    }
}

void NeuralNet::dump()
{
    printf("Neural Net with %ld layers\n", this->layers.size());
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        printf("Layer %ld - %ld units\n", i, this->operator[](i)->size());
    }
}
