#include <vector>
#include <math.h>

typedef float nn_num_t;

class Layer;

class NeuronLayer;

class Unit
{
    friend class Layer;

protected:
    std::vector<nn_num_t> connectionWeights;
    nn_num_t value_ = 0.0;

public:
    nn_num_t delta;
    nn_num_t error;

    nn_num_t raw() { return value_; };
    virtual nn_num_t activation() = 0;
    virtual nn_num_t activationDeriv() = 0;

    virtual void recalculate() = 0;

    void changeconnectionweight(int index, nn_num_t delta) { this->connectionWeights[index] += delta; };

    nn_num_t operator[](int index) { return connectionWeights[index]; };

    const std::vector<nn_num_t> &GetConnectionWeights() { return this->connectionWeights; };

    void addConnection(nn_num_t weight) { this->connectionWeights.push_back(weight); };
};

class Input : public Unit
{
public:
    Input() { value_ = 0.0; };
    virtual nn_num_t activation() { return value_; };
    virtual nn_num_t activationDeriv() { return 0.0; };
    void recalculate(){};
    void setValue(nn_num_t newValue) { this->value_ = newValue; };

    void backpropagate(nn_num_t error, nn_num_t alpha){};
};

class Neuron : public Unit
{
    friend class Layer;

private:
    Layer *inputLayer;

    void changeconnectionweight(int index, nn_num_t delta) { this->connectionWeights[index] += (this->connectionWeights[index] * delta); };

public:
    Neuron()
    {
        this->inputLayer = nullptr;
        this->delta = 0.0;
    };
    Neuron(Layer *inputLayer)
    {
        this->inputLayer = inputLayer;
        this->delta = 0.0;
    };

    void recalculate();

    void backpropagate(nn_num_t error, nn_num_t alpha);
};

class Logistic : public Neuron
{
public:
    Logistic(){};
    Logistic(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() { return 1.0 / (1.0 + exp(-1.0 * this->value_)); };
    nn_num_t activationDeriv()
    {
        nn_num_t a = this->activation();
        return a * (1.0 - a);
    };
};

class Perceptron : public Neuron
{
public:
    Perceptron(){};
    Perceptron(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() { return (this->value_ > 0.0) ? 1.0 : 0.0; };
    nn_num_t activationDeriv()
    {
        return (this->value_ == 0.0) ? MAXFLOAT : 0.0;
    };
};

class Linear : public Neuron
{
public:
    Linear(){};
    Linear(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() { return this->value_; };
    nn_num_t activationDeriv() { return 1.0; };
};

class BiasNeuron : public Unit
{
public:
    BiasNeuron() : Unit() { value_ = 1.0; };
    virtual nn_num_t activation() { return value_; };
    virtual nn_num_t activationDeriv() { return 0.0; };
    void recalculate(){};
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

    void BackPropagate(nn_num_t error){};

    std::vector<Unit *>::iterator begin() { return this->units.begin(); };

    std::vector<Unit *>::iterator end() { return this->units.end(); };
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
        logistic,
        perceptron,
        linear,
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

    void BackPropagate(nn_num_t expectedOutput);

    void UpdateWeights(nn_num_t learningRate);

    void dump();

    void setInput(nn_num_t value);

    void ForwardPropagate();
};
