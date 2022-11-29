#ifndef __NNET_H_
#define __NNET_H_

#include <math.h>
#include <vector>
#include <map>

#include "layers.h"

namespace SimpleNets
{
    class Unit;
    class Layer;

    class NeuralNet
    {
        friend class Layer;

    private:
        std::map<size_t, Unit *> units_;

    protected:
        // generate a unique ID for a new unit
        size_t AcquireNewUnitID();

        // generate a unit from the given type, ensuring it is handled correctly in the map of units
        // this is the only correct way to make new units for the net
        Unit *GenerateUnitFromType(neuronTypes t, size_t unitID);
        Unit *GenerateUnitFromType(neuronTypes t);

        const std::map<size_t, Unit *> &units();
        std::vector<Layer *> layers;

    public:
        // return the pointer to a given layer by index
        Layer *operator[](int index);

        // return the number of layers in the network
        size_t size();

        // return the size of a given layer by index
        size_t size(int index);


        // return the output of the network given the current input state
        // to be implemented by specific net types
        virtual nn_num_t Output() = 0;

        // print the network layers, units, and connections
        void Dump();

        // sets the input neuron values by index to the values provided
        void SetInput(const std::vector<nn_num_t> &values);

        // add a connection from a given ID to a given ID with the given weight
        void AddConnection(size_t fromId, size_t toId, nn_num_t w);
        
        // get the weight of a connection from a given ID to a given ID
        const nn_num_t GetWeight(size_t fromId, size_t toId);

        // change the weight of a connection from a given ID to a given ID by a delta
        void ChangeWeight(size_t fromId, size_t toId, nn_num_t delta);

        // set the weight of a connection from a given ID to a given ID to a value
        void SetWeight(size_t fromId, size_t toId, nn_num_t w);


        // void AddNeuron(size_t layer, neuronTypes t);
        // void RemoveNeuron(std::pair<size_t, size_t> index);
    };
} // namespace SimpleNets

#endif
