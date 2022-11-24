#include <stdio.h>
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
    nn_num_t delta = 0.0;
    nn_num_t error = 0.0;

    virtual ~Unit(){};

    nn_num_t raw() { return value_; };
    virtual nn_num_t activation() = 0;
    virtual nn_num_t activationDeriv() = 0;

    virtual void recalculate() = 0;

    void changeconnectionweight(int index, nn_num_t delta) { this->connectionWeights[index] += delta; };

    void setconnectionweight(int index, nn_num_t w) { this->connectionWeights[index] = w; };

    nn_num_t operator[](int index) { return connectionWeights[index]; };

    const std::vector<nn_num_t> &GetConnectionWeights() { return this->connectionWeights; };

    void addConnection(nn_num_t weight) { this->connectionWeights.push_back(weight); };
};

class Input : public Unit
{
public:
    explicit Input() { value_ = 0.0; };
    ~Input(){};
    nn_num_t activation() override { return value_; };
    nn_num_t activationDeriv() override { return 0.0; };
    void recalculate() override{};
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
    explicit Neuron(Layer *inputLayer) { this->inputLayer = inputLayer; };
    ~Neuron(){};

    void recalculate() override;

    void backpropagate(nn_num_t error, nn_num_t alpha);
};

class Logistic : public Neuron
{
public:
    explicit Logistic(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() override { return 1.0 / (1.0 + exp(-1.0 * this->value_)); };
    nn_num_t activationDeriv() override
    {
        nn_num_t a = this->activation();
        return a * (1.0 - a);
    };
};

class Perceptron : public Neuron
{
public:
    explicit Perceptron(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() override { return (this->value_ > 0) ? 1.0 : 0.0; };
    nn_num_t activationDeriv() override
    {
        nn_num_t a = 0.001 / (0.001 + exp(-100.0 * this->value_));
        return a * (1.0 - a);
    };
};

class Linear : public Neuron
{
public:
    explicit Linear(Layer *inputLayer) : Neuron(inputLayer){};
    nn_num_t activation() override { return this->value_; };
    nn_num_t activationDeriv() override { return 1.0; };
};

class BiasNeuron : public Unit
{
public:
    explicit BiasNeuron() : Unit() { value_ = 1.0; };
    virtual nn_num_t activation() override { return value_; };
    virtual nn_num_t activationDeriv() override { return 0.0; };
    void recalculate() override{};
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
    ~Layer()
    {
        for (auto u : this->units)
            delete u;
    };

    void AddUnit(Unit *u);

    std::size_t size() { return this->units.size(); };

    std::size_t index() { return this->index_; };

    std::vector<Unit *>::iterator begin() { return this->units.begin(); };

    std::vector<Unit *>::iterator end() { return this->units.end(); };
};

class NeuronLayer : public Layer
{
public:
    explicit NeuronLayer(NeuralNet *myNet_) : Layer(myNet_, true){};

    Neuron &operator[](int index) { return *static_cast<Neuron *>(this->units.at(index)); };
};

class InputLayer : public Layer
{
public:
    explicit InputLayer(NeuralNet *myNet_) : Layer(myNet_, false){};

    Input &operator[](int index) { return *static_cast<Input *>(this->units.at(index)); };
};

class OutputLayer : public Layer
{
public:
    explicit OutputLayer(NeuralNet *myNet_) : Layer(myNet_, false){};

    Neuron &operator[](int index) { return *static_cast<Neuron *>(this->units.at(index)); };
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
    NeuronLayer &hiddenLayer(int index) { return *static_cast<NeuronLayer *>((*this)[index]); };
    void BackPropagate(const std::vector<nn_num_t> &expectedOutput);
    void UpdateWeights(nn_num_t learningRate);
    void ForwardPropagate();

public:
    explicit NeuralNet(int nInputs);
    ~NeuralNet()
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            delete this->layers[i];
        }
    }
    std::size_t size() { return this->layers.size(); };
    std::size_t size(int index) { return (*this)[index]->size(); };

    void AddLayer(size_t size, enum neuronTypes t);

    void ConfigureOutput(int nOutputs, enum neuronTypes nt);

    nn_num_t Output();

    void Learn(const std::vector<nn_num_t> &expectedOutput, nn_num_t learningRate)
    {
        this->BackPropagate(expectedOutput);
        this->UpdateWeights(learningRate);
    };

    void dump();

    void setInput(const std::vector<nn_num_t> &values);

    const nn_num_t GetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to)
    {
        if ((from.first + 1 > this->size()) ||
            (from.second + 1 > this->size(from.first)) ||
            (from.first + 1 != to.first) ||
            (to.first + 1 > this->size()) ||
            (to.second + 1 > this->size(to.second)))
        {
            printf("Invalid request to get weight from layer %lu:%lu to layer %lu:%lu\n",
                   from.first, from.second, to.first, to.second);
            exit(1);
        }
        Layer &tl = *(*this)[to.first];
        return tl[to.second].GetConnectionWeights()[from.second];
    }

    void ChangeWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t delta)
    {
        if ((from.first + 1 > this->size()) ||
            (from.second + 1 > this->size(from.first)) ||
            (from.first + 1 != to.first) ||
            (to.first + 1 > this->size()) ||
            (to.second + 1 > this->size(to.second)))
        {
            printf("Invalid request to get weight from layer %lu:%lu to layer %lu:%lu\n",
                   from.first, from.second, to.first, to.second);
            exit(1);
        }
        Layer &tl = *(*this)[to.first];
        tl[to.second].changeconnectionweight(from.second, delta);
        // return tl[to.second].GetConnectionWeights()[from.second];
    }

    void SetWeight(std::pair<size_t, size_t> from, std::pair<size_t, size_t> to, nn_num_t w)
    {
        if ((from.first + 1 > this->size()) ||
            (from.second + 1 > this->size(from.first)) ||
            (from.first + 1 != to.first) ||
            (to.first + 1 > this->size()) ||
            (to.second + 1 > this->size(to.second)))
        {
            printf("Invalid request to get weight from layer %lu:%lu to layer %lu:%lu\n",
                   from.first, from.second, to.first, to.second);
            exit(1);
        }
        Layer &tl = *(*this)[to.first];
        tl[to.second].setconnectionweight(from.second, w);
        // return tl[to.second].GetConnectionWeights()[from.second];
    }
};
