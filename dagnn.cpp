#include "dagnn.h"
#include "queue"
namespace SimpleNets
{
    DAGNetwork::DAGNetwork(size_t nInputs,
                           std::vector<std::pair<neuronTypes, size_t>> hiddenNeurons,
                           std::pair<size_t, neuronTypes> outputFormat)
    {
        this->layers.push_back(new Layer(this, true));
        size_t i;
        for (i = 0; i < nInputs; i++)
        {
            this->layers.back()->AddUnit(this->GenerateUnitFromType(input));
        }

        Layer *hiddenLayer = new Layer(this, false);
        for (i = 0; i < hiddenNeurons.size(); i++)
        {
            hiddenLayer->AddUnit(this->GenerateUnitFromType(hiddenNeurons[i].first, hiddenNeurons[i].second));
        }

        // for(i = 0; i < hiddenNeurons.size(); i++)
        // {
        // }

        this->layers.push_back(hiddenLayer);
        /*for (i = 0; i < hiddenLayers.size(); i++)
        {
            Layer *l = new Layer(this, false);
            for (size_t j = 0; j < hiddenLayers[i].first; j++)
            {
                Unit *u = this->GenerateUnitFromType(hiddenLayers[i].second);
                for (auto k = this->layers.back()->begin(); k != this->layers.back()->end(); ++k)
                {
                    u->AddConnection(*k, 0.1);
                }
                l->AddUnit(u);
            }
            this->layers.push_back(l);
        }*/

        Layer *ol = new Layer(this, false);
        for (size_t j = 0; j < outputFormat.first; j++)
        {
            ol->AddUnit(this->GenerateUnitFromType(outputFormat.second));
        }
        this->layers.push_back(ol);
    }

    DAGNetwork::~DAGNetwork()
    {
    }

    void DAGNetwork::Recalculate()
    {
        std::set<Unit *> touched;
        std::queue<Unit *> toRecalculate;
        for (auto u = this->layers.back()->begin(); u != this->layers.back()->end(); ++u)
        {
            toRecalculate.push((*u));
        }
        while (toRecalculate.size() > 0)
        {
            Unit *u = toRecalculate.front();
            if (touched.count(u) == 0)
            {
                // if we have reached an input, nothing to do
                if (u->GetConnections().size() == 0)
                {
                    toRecalculate.pop();
                }
                // otherwise, push all this node's inputs in front of it in the queue
                else
                {
                    for (auto c : u->GetConnections())
                    {
                        toRecalculate.push(c.from.u);
                    }
                }
            }
            else
            {
                u->CalculateValue();
                toRecalculate.pop();
            }
            touched.insert(u);
        }
    }

    void DAGNetwork::Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate)
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

    nn_num_t DAGNetwork::Output()
    {
        this->Recalculate();
        switch (this->layers.back()->size())
        {
        case 1:
            return ((*this->layers.back())[0].Activation());
            break;

        default:
            int maxIndex = 0;
            nn_num_t maxValue = -1.0 * MAXFLOAT;
            Layer &ol = *this->layers.back();
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

    void DAGNetwork::BackPropagate(const std::vector<nn_num_t> &expectedOutput)
    {
        for (auto u : this->units())
        {
            u.second->delta = 0.0;
        }
        // delta of each output j = activation derivative(j) * (expected(j) - actual(j))
        Layer &ol = *this->layers.back();
        std::set<Unit *> visited;
        std::queue<Unit *> toPropagate;
        for (size_t j = 0; j < ol.size(); j++)
        {
            ol[j].delta = ol[j].ActivationDeriv() * (expectedOutput[j] - ol[j].Activation());
            toPropagate.push(&ol[j]);

        }
        while (toPropagate.size() > 0)
        {
            Unit *j = toPropagate.front();
            toPropagate.pop();
            if (visited.count(j) == 0)
            {
                visited.insert(j);
                for (auto c : j->GetConnections())
                {
                    Unit *i = c.from.u;
                    printf("use %lu->%lu connection to build %lu's delta\n", i->Id(), j->Id(), i->Id());
                    i->delta += (c.weight * j->delta) * i->ActivationDeriv();
                    toPropagate.push(i);
                }
            }
        }
    }

    void DAGNetwork::UpdateWeights(nn_num_t learningRate)
    {
        for(auto u : this->units())
        {
            Unit *j = u.second;
            for(auto v : j->GetConnections())
            {
                Unit *i = v.from.u;
                printf("update %lu->%lu weight\n", i->Id(), j->Id());
                j->ChangeConnectionWeight(i, learningRate * i->Activation() * j->delta);
            }
        }
    }
} // namespace SimpleNets
