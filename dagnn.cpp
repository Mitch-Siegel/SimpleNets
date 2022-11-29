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
            Unit *u = this->GenerateUnitFromType(outputFormat.second);
            for (auto k = this->layers.back()->begin(); k != this->layers.back()->end(); ++k)
            {
                u->AddConnection(*k, 0.05);
            }
            ol->AddUnit(u);
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
        for (size_t j = 0; j < ol.size(); j++)
        {
            ol[j].delta = ol[j].ActivationDeriv() * (expectedOutput[j] - ol[j].Activation());
        }

        std::queue<Unit *> toPropagate;
        for (auto u = this->layers.back()->begin(); u != this->layers.back()->end(); ++u)
        {
            toPropagate.push((*u));
        }
        while (toPropagate.size() > 0)
        {
            Unit *u = toPropagate.front();
            u->delta *= u->ActivationDeriv();
            toPropagate.pop();
            for (auto c : u->GetConnections())
            {
                c.from.u->delta += c.weight * u->delta;
                toPropagate.push(c.from.u);
            }
        }
        /*
        // for all other layers, delta of a node i in the layer is:
        // activation derivative(i) * sum for all j(weight of connection from i to j * delta(j))
        for (size_t li = this->layers.size() - 1; li > 0; --li)
        {
            Layer &l = *this->layers[li - 1];
            Layer &nl = *this->layers[li];
            for (size_t i = 0; i < l.size(); i++)
            {
                nn_num_t sum = 0.0;
                for (size_t j = 0; j < nl.size(); j++)
                {
                    std::set<Connection> connections = nl[j].GetConnections();
                    sum += (*connections.find(Connection(&l[i]))).weight * nl[j].delta;
                }
                l[i].delta = sum * l[i].ActivationDeriv();
            }
        }
        */
    }

    void DAGNetwork::UpdateWeights(nn_num_t learningRate)
    {
        for (size_t li = this->size() - 1; li > 0; li--)
        {
            Layer &l = *(*this)[li];
            Layer &pl = *(*this)[li - 1];
            for (auto j = l.begin(); j != l.end(); ++j)
            {
                for (auto i = pl.begin(); i != pl.end(); ++i)
                {
                    (*j)->ChangeConnectionWeight(*i, learningRate * (*i)->Activation() * (*j)->delta);
                }
            }
        }
    }
} // namespace SimpleNets
