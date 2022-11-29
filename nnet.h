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
        std::map<size_t, Unit *> units;

    protected:
        size_t AcquireNewUnitID();

        Unit *GenerateUnitFromType(neuronTypes t, size_t unitID);
        Unit *GenerateUnitFromType(neuronTypes t);

        std::vector<Layer *> layers;

    public:
        Layer *operator[](int index);
        size_t size();
        size_t size(int index);
        virtual nn_num_t Output() = 0;

        void Dump();

        void SetInput(const std::vector<nn_num_t> &values);

        const nn_num_t GetWeight(size_t fromId, size_t toId);
        void ChangeWeight(size_t fromId, size_t toId, nn_num_t delta);
        void SetWeight(size_t fromId, size_t toId, nn_num_t w);

        // void AddNeuron(size_t layer, neuronTypes t);
        // void RemoveNeuron(std::pair<size_t, size_t> index);
    };
} // namespace SimpleNets

#endif
