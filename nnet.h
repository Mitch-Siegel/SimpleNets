#include <vector>

typedef float nn_num_t;

class Unit
{
protected:
    std::vector<nn_num_t> connectionWeights;
    nn_num_t value;

public:
    nn_num_t delta;

    virtual nn_num_t output() = 0;

    void changeconnectionweight(int index, nn_num_t delta) { this->connectionWeights[index] += (this->connectionWeights[index] * delta); };

    virtual void backpropagate(nn_num_t error, nn_num_t alpha) = 0;
};

class Input : public Unit
{
public:
    Input() { value = 0.0; };
    nn_num_t output() { return this->value; }
    void setValue(nn_num_t newValue) { this->value = newValue; };

    void backpropagate(nn_num_t error, nn_num_t alpha){};
};

class Layer;

class NeuronLayer;

class Neuron : public Unit
{
    friend class Layer;

private:
    bool needRecompute;

    Layer *inputLayer;

    virtual nn_num_t activation(nn_num_t) = 0;

public:
    Neuron();
    Neuron(Layer *inputLayer);

    nn_num_t output();

    void backpropagate(nn_num_t error, nn_num_t alpha);
};

class Perceptron : public Neuron
{
private:
    nn_num_t activation(nn_num_t input);

public:
    Perceptron();
    Perceptron(Layer *inputLayer);
    void learn(std::vector<nn_num_t> inputs, nn_num_t output, nn_num_t alpha);
};

class BiasNeuron : public Unit
{
public:
    BiasNeuron() : Unit(){};
    nn_num_t output() { return 1.0; }
    void backpropagate(nn_num_t error, nn_num_t alpha){};
};

class NeuralNet;
class Layer
{
    friend class NeuronLayer;
    friend class OutputLayer;

private:
    std::size_t index_;
    std::vector<Unit *> units;
    NeuralNet *myNet;

public:
    Unit *operator[](int index) { return this->units.at(index); };

    Layer(NeuralNet *myNet_, bool addBias);

    void AddUnit(Unit *u);

    std::size_t size() { return this->units.size(); };

    std::size_t index() { return this->index_; };

    void BackPropagate(nn_num_t error, nn_num_t alpha){};

    void StartBackProp();
};

class NeuronLayer : public Layer
{
public:
    Neuron *operator[](int index) { return static_cast<Neuron *>(this->units.at(index)); };

    NeuronLayer(NeuralNet *myNet_) : Layer(myNet_, true){};
};

class OutputLayer : public Layer
{
public:
    Neuron *operator[](int index) { return static_cast<Neuron *>(this->units.at(index)); };

    OutputLayer(NeuralNet *myNet_) : Layer(myNet_, false){};
};

class NeuralNet
{
public:
    enum neuronTypes
    {
        perceptron,
    };

    enum outputTypes
    {
        output_singular,
        output_null,
    };

private:
    std::vector<Layer *> layers;
    enum outputTypes outputType = output_null;

    void AddOutputLayer(int size, enum neuronTypes t);

public:
    NeuralNet(int nInputs);

    std::size_t size() { return this->layers.size(); };

    Layer *operator[](int index) { return this->layers[index]; };

    void AddLayer(int size, enum neuronTypes t);

    void ConfigureOutput(enum outputTypes t, enum neuronTypes nt);

    nn_num_t Output();

    void BackPropagate(nn_num_t expectedOutput, nn_num_t alpha);

    void dump();
};
