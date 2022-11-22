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

void Neuron::recalculate()
{
    this->value_ = 0.0;
    std::size_t nInputs = this->connectionWeights.size();
    for (std::size_t i = 0; i < nInputs; i++)
    {
        this->value_ += (this->inputLayer->operator[](i)->output() * this->connectionWeights[i]);
    }
    this->value_ = this->activation(this->value_);
}

Perceptron::Perceptron(Layer *inputLayer) : Neuron(inputLayer)
{
}

Perceptron::Perceptron() : Neuron()
{
}

nn_num_t Perceptron::activation(nn_num_t input)
{
    return 1.0 / (1.0 + exp(-1.0 * input));
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

    if (this->index_ < this->myNet->size() - 1)
    {
        NeuronLayer *forwardLayer = static_cast<NeuronLayer *>(this->myNet->operator[](this->index_ + 1));
        std::size_t forwardLayerSize = forwardLayer->size();
        for (std::size_t i = 0; i < forwardLayerSize; i++)
        {
            forwardLayer->operator[](i)->addConnection(0.01);
        }
    }
}

NeuralNet::NeuralNet(int nInputs)
{
    this->layers.push_back(new Layer(this, false));
    for (int i = 0; i < nInputs; i++)
    {
        this->layers.back()->AddUnit(new Input());
    }
    // this->layers.push_back(Layer(this));
}

void NeuralNet::AddLayer(int size, enum neuronTypes t)
{
    NeuronLayer *newLayer = new NeuronLayer(this);
    for (int i = 0; i < size; i++)
    {
        switch (t)
        {
        case perceptron:
            newLayer->AddUnit(new Perceptron(this->layers.back()));
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
        case perceptron:
            ol->AddUnit(new Perceptron(this->layers.back()));
            break;
        }
    }
    this->layers.push_back(ol);
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

void NeuralNet::BackPropagate(nn_num_t expectedOutput)
{
    /*
    def backward_propagate_error(network, expected):
     for i in reversed(range(len(network))):
         layer = network[i]
         errors = list()
         if i != len(network)-1:
             for j in range(len(layer)):
                 error = 0.0
                 for neuron in network[i + 1]:
                     error += (neuron['weights'][j] * neuron['delta'])
                 errors.append(error)
         else:
             for j in range(len(layer)):
                 neuron = layer[j]
                 errors.append(neuron['output'] - expected[j])
         for j in range(len(layer)):
             neuron = layer[j]
             neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
             */

    /*

    def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        for j in range(len(layer)):
            fromNeuron = layer[j]
            error = 0.0
            if i != len(network)-1:                           #This identifies all but the last (output) layer
                for toNeuron in network[i + 1]:
                    error += (toNeuron['weights'][j] * toNeuron['delta'])
            else:                                             #This is the last (output) layer
                error = expected[j] - fromNeuron['output']
            fromNeuron['error'] = error
            fromNeuron['delta'] = error * transfer_derivative(fromNeuron['output'])*/

    for (size_t i = this->layers.size(); i > 0;)
    {
        --i;
        printf("backprop layer %lu\n", i);
        Layer *layer = this->layers[i];

        for (size_t j = 0; j < layer->size(); j++)
        {
            Unit *from = layer->operator[](j);
            nn_num_t error = 0.0;
            if (i < this->layers.size() - 1)
            {
                for (auto to = this->layers[i + 1]->begin(); to != this->layers[i + 1]->end(); ++to)
                {
                    error += (*to)->GetConnectionWeights()[j] * (*to)->delta;
                }
                // from->delta = error * (from->output() * (1.0 - from->output()));
            }
            else
            {
                error = expectedOutput - from->output();
                // from->delta = error;
            }
            from->error = error;
            from->delta = error * (from->output() * (1.0 - from->output()));
        }

        /*std::vector<nn_num_t> errors;
        if (i != this->layers.size() - 1)
        {
            for (size_t j = 0; j < layer->size(); j++)
            {
                nn_num_t error = 0.0;
                Layer *nextLayer = this->layers.operator[](i + 1);
                for (auto n = nextLayer->begin(); n != nextLayer->end(); ++n)
                {
                    error += (*n)->operator[](j) * (*n)->delta;
                }
                errors.push_back(error);
            }
        }
        else
        {
            for (size_t j = 0; j < layer->size(); j++)
            {
                Unit *n = layer->operator[](j);
                errors.push_back(n->output() - expectedOutput);
            }
        }
        printf("Errors[%lu] = {", i);
        for (nn_num_t n : errors)
        {
            printf("% .2f, ", n);
        }
        printf("}\n");
        for (size_t j = 0; j < layer->size(); j++)
        {
            Unit *n = layer->operator[](j);
            printf("delta %lu,%lu is % .2f\n", layer->index(), j, errors[j] * (n->output() * (1.0 - n->output())));
            n->delta = errors[j] * (n->output() * (1.0 - n->output()));
        }*/
    }

    // size_t networkSize = this->size();
    // for(int i = networkSize - 1; i > 0; i--)
    // {
    // Layer *l = this->operator[](i);
    // std::vector<nn_num_t> errors(l->size());
    // if(i != networkSize - 1)
    // {
    // size_t layerSize = l->size();
    // Layer *nextLayer = this->operator[](i + 1);
    // for(int j = 0; j < layerSize; j++)
    // {
    // nn_num_t error = 0.0;
    // for()
    // }
    // }
    // }
}

void NeuralNet::UpdateWeights(nn_num_t learningRate)
{
    /*
        for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']
            */
    std::vector<nn_num_t> inputs;
    for (size_t i = 1; i < this->size(); i++)
    {
        inputs.clear();
        // std::vector<nn_num_t> inputs;
        // if (i > 0)
        // {
        printf("Layer %lu: ", i);
        for (auto n = this->operator[](i - 1)->begin(); n != this->operator[](i - 1)->end(); ++n)
        {
            inputs.push_back((*n)->output());
            printf("% .2f, ", inputs.back());
        }
        printf("\n");
        for (auto n = this->operator[](i)->begin(); n != this->operator[](i)->end(); ++n)
        {
            for (size_t j = 1; j < inputs.size(); j++)
            {
                printf("changing connection weight for %lu,%lu by % .2f\n", i, j, learningRate * (*n)->delta * inputs[j]);
                (*n)->changeconnectionweight(j, learningRate * (*n)->delta * inputs[j]);
            }
            (*n)->changeconnectionweight(0, learningRate * (*n)->delta);
        }
        // }
    }
}

void NeuralNet::dump()
{
    printf("Neural Net with %ld layers\n", this->layers.size());
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        Layer *l = this->operator[](i);
        printf("Layer %ld - %ld units\n", i, l->size());
        for (size_t j = 0; j < l->size(); j++)
        {
            Unit *u = l->operator[](j);
            printf("Neuron %2lu: activation: % .2f, delta % .2f, error % .2f, weights:\n\t", j, u->output(), u->delta, u->error);
            auto cw = u->GetConnectionWeights();
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

void NeuralNet::setInput(nn_num_t value)
{
    Input *i = static_cast<Input *>(this->layers[0]->operator[](0));
    i->setValue(value);
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
