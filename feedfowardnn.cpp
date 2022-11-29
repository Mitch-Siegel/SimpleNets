#include "feedforwardnn.h"
namespace SimpleNets
{
    FeedForwardNeuralNet::FeedForwardNeuralNet(size_t nInputs, std::vector<std::pair<size_t, neuronTypes>> hiddenLayers, std::pair<size_t, neuronTypes> outputFormat)
    {
        this->layers.push_back(new Layer(this, true));
        size_t i;
        for (i = 0; i < nInputs; i++)
        {
            this->layers.back()->AddUnit(this->GenerateUnitFromType(input));
        }

        for (i = 0; i < hiddenLayers.size(); i++)
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
        }

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

    FeedForwardNeuralNet::~FeedForwardNeuralNet()
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            delete this->layers[i];
        }
    }

    void FeedForwardNeuralNet::Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate)
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

    nn_num_t FeedForwardNeuralNet::Output()
    {
        this->ForwardPropagate();
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

    void FeedForwardNeuralNet::BackPropagate(const std::vector<nn_num_t> &expectedOutput)
    {
        // delta of each output j = activation derivative(j) * (expected(j) - actual(j))
        Layer &ol = *this->layers.back();
        for (size_t j = 0; j < ol.size(); j++)
        {
            ol[j].delta = ol[j].ActivationDeriv() * (expectedOutput[j] - ol[j].Activation());
        }

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
    }

    void FeedForwardNeuralNet::UpdateWeights(nn_num_t learningRate)
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

    void FeedForwardNeuralNet::ForwardPropagate()
    {
        for (size_t i = 1; i < this->size(); i++)
        {
            Layer *l = this->operator[](i);
            for (auto u = l->begin(); u != l->end(); ++u)
            {
                (*u)->CalculateValue();
            }
        }
    }
} // namespace SimpleNets
