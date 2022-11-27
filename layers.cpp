#include "layers.h"
#include "nnet.h"

Unit &Layer::operator[](size_t index)
{
    return *this->units.at(index);
};

Layer::Layer(NeuralNet *myNet_, bool addBias)
{
    this->myNet = myNet_;
    this->index_ = myNet_->size();

    if (addBias)
    {
        this->AddUnit(new BiasNeuron());
    }
}

Layer::~Layer()
{
    for (auto u : this->units)
        delete u;
}

void Layer::AddUnit(Unit *u)
{
    this->units.push_back(u);
    if (this->index_ > 0)
    {
        for (size_t i = 0; i < this->myNet->operator[](this->index_ - 1)->size(); i++)
        {
            u->AddConnection(0.01);
        }
    }

    if (this->index_ + 1 < this->myNet->size())
    {
    
        NeuronLayer &forwardLayer = *static_cast<NeuronLayer *>((*this->myNet)[this->index_ + 1]);
        std::size_t forwardLayerSize = forwardLayer.size();
        for (std::size_t i = 0; i < forwardLayerSize; i++)
        {
            forwardLayer[i].AddConnection(0.01);
        }
    }
}

void Layer::RemoveUnit(size_t index)
{
    delete this->units[index];
    this->units.erase(this->units.begin() + index);
    Layer *nl = (*this->myNet)[this->index_ + 1];
    for (size_t i = 0; i < nl->size(); i++)
    {
        (*nl)[i].RemoveConnection(index);
    }
}

size_t Layer::size()
{
    return this->units.size();
}

size_t Layer::Index()
{
    return this->index_;
}

void Layer::SetIndex(size_t i)
{
    this->index_ = i;
}

std::vector<Unit *>::iterator Layer::begin()
{
    return this->units.begin();
}

std::vector<Unit *>::iterator Layer::end()
{
    return this->units.end();
}

// NeuronLayer
NeuronLayer::NeuronLayer(NeuralNet *myNet_, bool addBias) : Layer(myNet_, addBias){};

Neuron &NeuronLayer::operator[](int index)
{
    return *static_cast<Neuron *>(this->units.at(index));
}

void NeuronLayer::SetInputLayer(Layer *l)
{
    for (auto u = this->begin(); u != this->end(); ++u)
    {
        Neuron *n = static_cast<Neuron *>(*u);
        n->SetInputLayer(l);
    };
}

// InputLayer

InputLayer::InputLayer(NeuralNet *myNet_) : Layer(myNet_, false)
{
}

Input &InputLayer::operator[](int index)
{
    return *static_cast<Input *>(this->units.at(index));
}

// OutputLayer
OutputLayer::OutputLayer(NeuralNet *myNet_) : NeuronLayer(myNet_, false)
{
}

Neuron &OutputLayer::operator[](int index)
{
    return *static_cast<Neuron *>(this->units.at(index));
}
