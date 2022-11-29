#include "nnet.h"

namespace SimpleNets
{
    class DAGNetwork : public NeuralNet
    {

    private:
        void BackPropagate(const std::vector<nn_num_t> &expectedOutput);
        void UpdateWeights(nn_num_t learningRate);
        void Recalculate();

    public:
        DAGNetwork(size_t nInputs,
                   std::vector<std::pair<neuronTypes, size_t>> hiddenNeurons,
                   std::pair<size_t, neuronTypes> outputFormat);
        ~DAGNetwork();

        void AddLayer(size_t size, enum neuronTypes t);

        void Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate);

        nn_num_t Output() override;
    };
} // namespace SimpleNets
