#include <math.h>
#include <stdio.h>

#include "nnet.h"
namespace SimpleNets
{
    // Neural Net base

    size_t NeuralNet::AcquireNewUnitID()
    {
        if (this->units_.size() == 0 || (this->units_.size() == (this->units_.rbegin()->second->Id() + 1)))
        {
            return this->units_.size();
        }
        else
        {
            size_t s = this->units_.size();
            for (size_t i = 0; i < s; i++)
            {
                if (this->units_.count(i) == 0)
                {
                    return i;
                }
            }
            printf("Error - expecteed gap in units map but couldn't find one!\n");
            exit(1);
        }
    }

    Unit *NeuralNet::GenerateUnitFromType(neuronTypes t, size_t id)
    {
        Unit *u;
        switch (t)
        {
        case input:
            u = new Units::Input(id);
            break;

        case bias:
            u = new Units::BiasNeuron(id);
            break;

        case logistic:
            u = new Units::Logistic(id);
            break;

        case perceptron:
            u = new Units::Perceptron(id);
            break;

        case linear:
            u = new Units::Linear(id);
            break;
        }
        this->units_[id] = u;
        return u;
    }
    Unit *NeuralNet::GenerateUnitFromType(neuronTypes t)
    {
        return this->GenerateUnitFromType(t, this->AcquireNewUnitID());
    }

    const std::map<size_t, Unit *> &NeuralNet::units()
    {
        return this->units_;
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
                printf("Neuron %2lu (type %10s): raw: %f, delta %f, error %f\n\tactivation: %f\n\tweights:", u.Id(), GetNeuronTypeName(u.type()), u.Raw(), u.delta, u.error, u.Activation());
                for (auto connection : u.GetConnections())
                {
                    printf("%lu->this % 0.07f, ", connection.from.u->Id(), connection.weight);
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
        for (size_t i = 0; i < values.size(); i++)
        {
            // offset by 1 to skip over bias neuron at index 0
            Units::Input *input = static_cast<Units::Input *>(*(this->layers[0]->begin() + 1 + i));
            input->SetValue(values[i]);
        }
    }

    void NeuralNet::AddConnection(size_t fromId, size_t toId, nn_num_t w)
    {
        if(this->units_.count(fromId) == 0)
        {
            printf("Error generating connection from %lu->%lu - source unit doesn't exist!\n", fromId, toId);
        }

        if(this->units_.count(toId) == 0)
        {
            printf("Error generating connection from %lu->%lu - destination unit doesn't exist!\n", fromId, toId);
        }
        this->units_[toId]->AddConnection(this->units_[fromId], w);
    }

    const nn_num_t NeuralNet::GetWeight(size_t fromId, size_t toId)
    {
        Unit *from = this->units_[fromId];
        return from->GetConnections().find(Connection(toId))->weight;
    }

    /*
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
    */
} // namespace SimpleNets
