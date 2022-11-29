#ifndef __UNITS_H_
#define __UNITS_H_

#include <set>

#include "connection.h"

namespace SimpleNets
{

    // enum for the different types of neurons
    enum neuronTypes
    {
        input,
        bias,
        logistic,
        perceptron,
        linear,
    };

    // return the cstr for the name of a given type in the enum
    const char *GetNeuronTypeName(neuronTypes t);

    class Layer;

    /*
     * A unit represents a node in the network
     * connections flow from left to right (lower to higher layer indices)
     */
    class Unit
    {
    // baseline read-only identifying information about the unit
    private:
        size_t id_;
        neuronTypes type_;

    protected:
        // list of all connections starting from a different unit and going to this unit
        std::set<Connection> connections;

        // internal value of the unit, calculated based on what type it is
        nn_num_t value_ = 0.0;

    public:
        // parameters for learning
        nn_num_t delta = 0.0;
        nn_num_t error = 0.0;

        Unit(size_t id, neuronTypes type);
        virtual ~Unit() = 0;
        
        // getters for read-only info
        const size_t Id();
        const neuronTypes type();
        
        // returns the raw value of the unit
        nn_num_t Raw();
        // returns the output of the unit based on its activation function
        virtual nn_num_t Activation() = 0;

        // returns the derivative of the unit's activation function at its current value
        virtual nn_num_t ActivationDeriv() = 0;

        // updates the value of the unit based on the current values of its connections
        virtual void CalculateValue() = 0;

        // alter the weight of a connection from a given unit to this unit
        void ChangeConnectionWeight(Unit *from, nn_num_t delta);
        void SetConnectionWeight(Unit *from, nn_num_t w);

        // get all connections to this unit
        const std::set<Connection> &GetConnections();
        

        // add a connection from a given unit to this unit with weight w
        void AddConnection(Unit *u, nn_num_t w);

        // remove the connection from a given unit to this unit
        void RemoveConnection(Unit *u);

        // remove all connection from any units to this unit
        void Disconnect();
    };

    namespace Units
    {
        class Input : public Unit
        {
        public:
            explicit Input(size_t id);
            ~Input(){};

            nn_num_t Activation() override;
            nn_num_t ActivationDeriv() override;
            void CalculateValue() override;
            void SetValue(nn_num_t newValue);
        };

        class Neuron : public Unit
        {
            friend class Layer;
            friend class NeuronLayer;
            friend class OutputLayer;

        public:
            explicit Neuron(size_t id, neuronTypes type);
            ~Neuron();

            void CalculateValue() override;
        };

        class Logistic : public Neuron
        {
        public:
            explicit Logistic(size_t id);
            nn_num_t Activation() override;
            nn_num_t ActivationDeriv() override;
        };

        class Perceptron : public Neuron
        {
        public:
            explicit Perceptron(size_t id);
            nn_num_t Activation() override;
            nn_num_t ActivationDeriv() override;
        };

        class Linear : public Neuron
        {
        public:
            explicit Linear(size_t id);
            nn_num_t Activation() override;
            nn_num_t ActivationDeriv() override;
        };

        class BiasNeuron : public Unit
        {
        public:
            explicit BiasNeuron(size_t id);
            nn_num_t Activation() override;
            nn_num_t ActivationDeriv() override;
            void CalculateValue() override;
        };
    } // namespace Units

} // namespace SimpleNets

#endif
