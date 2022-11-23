#include <math.h>

#include "nnet.h"
#include <stdio.h>

void Neuron::recalculate()
{
    this->value_ = 0.0;
    std::size_t nInputs = this->connectionWeights.size();
    for (std::size_t i = 0; i < nInputs; i++)
    {
        this->value_ += (this->inputLayer->operator[](i)->activation() * this->connectionWeights[i]);
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
    this->layers.push_back(new Layer(this, true));
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
        return (this->layers.back()->operator[](0)->activation());
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
    // for each node j in the output layer do
    //     Delta[j] <- g'(in_j) \times (y_j - a_j)
    // for l = L-1 to 1 do
    //     for each node i in layer l do
    //         Delta[i] <- g'(in_i) * \sum_j w_ij Delta[j]
    // for each weight w_ij in network do
    //     w_ij <- w_ij + alpha * a_i * delta_j
    */
    for (auto j = this->layers.back()->begin(); j != this->layers.back()->end(); ++j)
    {
        (*j)->delta = (*j)->activationDeriv() * (expectedOutput - (*j)->activation());
        printf("output delta is %f\n", (*j)->activationDeriv() * (expectedOutput - (*j)->activation()));
    }

    // for l = L-1 to 1
    // for(auto l = this->layers.rbegin() + 1; l != this->layers.rend(); ++l)
    for (size_t li = this->layers.size() - 2; li > 0; li--)
    {
        Layer *l = this->layers[li];
        Layer *nl = this->layers[li + 1];
        // for each node i in layer l
        for (size_t i = 0; i < l->size(); i++)
        {
            nn_num_t sum = 0.0;
            for (size_t nli = 0; nli < nl->size(); nli++)
            {
                Unit *j = nl->operator[](nli);
                sum += (j->GetConnectionWeights()[i] * j->delta);
            }
            l->operator[](i)->delta = sum * l->operator[](i)->activationDeriv();

            // (*i)->delta =
        }
    }
    /*
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
         */
}

void NeuralNet::UpdateWeights(nn_num_t learningRate)
{
    printf("updating weights with learning rate %f\n", learningRate);
    for (size_t li = this->size() - 1; li > 0; li--)
    {
        Layer *l = this->operator[](li);
        Layer *pl = this->operator[](li - 1);
        for (auto j = l->begin(); j != l->end(); ++j)
        {
            for (size_t i = 0; i < pl->size(); i++)
            {
                (*j)->changeconnectionweight(i, learningRate * pl->operator[](i)->activation() * (*j)->delta);
            }
        }
    }

    /*std::vector<nn_num_t> inputs;
    for (size_t i = 1; i < this->size(); i++)
    {
        inputs.clear();
        // std::vector<nn_num_t> inputs;
        // if (i > 0)
        // {
        printf("Layer %lu: ", i);
        for (auto n = this->operator[](i - 1)->begin(); n != this->operator[](i - 1)->end(); ++n)
        {
            inputs.push_back((*n)->activation());
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
    }*/
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
            printf("Neuron %2lu: raw: % .8f, delta % .8f, error % .8f\n\tactivation: %f\n\tweights:", j, u->raw(), u->delta, u->error, u->activation());
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
