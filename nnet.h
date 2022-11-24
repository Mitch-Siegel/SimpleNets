#include <vector>
#include <math.h>

typedef float nn_num_t;

class Layer;

class NeuronLayer;

/*
 * A unit represents a node in the network
 * connections flow from left to right (lower to higher layer indices)
 */
class Unit
{
    friend class Layer;

protected:
    std::vector<nn_num_t> connectionWeights;
    nn_num_t value_ = 0.0;

public:
    nn_num_t delta;
    nn_num_t error;

    // virtual Unit() = 0;
    virtual ~Unit(){};

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
    ~Input(){};
    nn_num_t activation() { return value_; };
    nn_num_t activationDeriv() { return 0.0; };
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
    // Neuron()
    // {
    // this->inputLayer = nullptr;
    // this->delta = 0.0;
    // };
    Neuron(Layer *inputLayer)
    {
        this->inputLayer = inputLayer;
        this->delta = 0.0;
    };
    ~Neuron(){};

    void recalculate();

    void backpropagate(nn_num_t error, nn_num_t alpha);
};

class Logistic : public Neuron
{
public:
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
    Perceptron(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() { return (this->value_ > 0) ? 1.0 : 0.0; };
    nn_num_t activationDeriv()
    {
        nn_num_t a = 0.001 / (0.001 + exp(-100.0 * this->value_));
        return a * (1.0 - a);
    };
};

class Linear : public Neuron
{
public:
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
    friend class InputLayer;
    friend class OutputLayer;

private:
    std::size_t index_;
    std::vector<Unit *> units;
    NeuralNet *myNet;

public:
    Unit &operator[](int index) { return *this->units.at(index); };

    Layer(NeuralNet *myNet_, bool addBias);
    ~Layer();

    void AddUnit(Unit *u);

    std::size_t size() { return this->units.size(); };

    std::size_t index() { return this->index_; };

    std::vector<Unit *>::iterator begin() { return this->units.begin(); };

    std::vector<Unit *>::iterator end() { return this->units.end(); };
};

class NeuronLayer : public Layer
{
public:
    Neuron &operator[](int index) { return *static_cast<Neuron *>(this->units.at(index)); };

    NeuronLayer(NeuralNet *myNet_) : Layer(myNet_, true){};
};

class InputLayer : public Layer
{
public:
    Input &operator[](int index) { return *static_cast<Input *>(this->units.at(index)); };

    InputLayer(NeuralNet *myNet_) : Layer(myNet_, false){};
};

class OutputLayer : public Layer
{
public:
    Neuron &operator[](int index) { return *static_cast<Neuron *>(this->units.at(index)); };

    OutputLayer(NeuralNet *myNet_) : Layer(myNet_, false){};
};

class NeuralNet
{

    friend class Layer;

public:
    enum neuronTypes
    {
        logistic,
        perceptron,
        linear,
    };

private:
    std::vector<Layer *> layers;
    int nOutputs = 0;

    void AddOutputLayer(int size, enum neuronTypes t);
    Layer *operator[](int index) { return this->layers[index]; };
    OutputLayer &output() { return *static_cast<OutputLayer *>(this->layers.back()); };
    NeuronLayer &hiddenLayer(int index) { return *static_cast<NeuronLayer *>((*this)[index]); };
    void BackPropagate(std::vector<nn_num_t> expectedOutput);
    void UpdateWeights(nn_num_t learningRate);
    void ForwardPropagate();


public:
    NeuralNet(int nInputs);

    std::size_t size() { return this->layers.size(); };

    void AddLayer(int size, enum neuronTypes t);

    void ConfigureOutput(int nOutputs, enum neuronTypes nt);

    nn_num_t Output();

    void Learn(std::vector<nn_num_t> expectedOutput, nn_num_t learningRate)
    {
        this->BackPropagate(expectedOutput);
        this->UpdateWeights(learningRate);
    };

    void dump();

    void setInput(std::vector<nn_num_t> values);

};
