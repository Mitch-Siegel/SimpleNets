#include "nnet.h"

class FeedForwardNeuralNet : public NeuralNet
{

private:
    int nOutputs = 0;
    void AddOutputLayer(int size, enum neuronTypes t);
    void BackPropagate(const std::vector<nn_num_t> &expectedOutput);
    void UpdateWeights(nn_num_t learningRate);
    void ForwardPropagate();

public:
    explicit FeedForwardNeuralNet(int nInputs);
    ~FeedForwardNeuralNet();

    void AddLayer(size_t size, enum neuronTypes t);

    void ConfigureOutput(int nOutputs, enum neuronTypes nt);

    void Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate);

    nn_num_t Output() override;
};