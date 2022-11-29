#include "units.h"
#include "layers.h"

Connection::Connection(Unit *u, nn_num_t weight)
{
    this->from.u = u;
    this->weight = weight;
    this->idOnly = false;
}

Connection::Connection(Unit *u)
{
    this->from.u = u;
    this->weight = 0.0;
    this->idOnly = false;
}

Connection::Connection(size_t id)
{
    this->from.id = id;
    this->weight = 0.0;
    this->idOnly = true;
}

bool Connection::operator==(const Connection &b)
{
    return (this->from.u == b.from.u) || (this->from.id == b.from.id);
}

bool Connection::operator<(const Connection &b) const
{
    if (this->idOnly)
    {
        return this->from.id < b.from.id;
    }
    else
    {
        return this->from.u->Id() < b.from.u->Id();
    }
}

// Unit

const char *GetNeuronTypeName(neuronTypes t)
{
    switch (t)
    {
    case input:
        return "Input";
        break;

    case bias:
        return "Bias";
        break;

    case logistic:
        return "Logistic";
        break;

    case perceptron:
        return "Perceptron";
        break;

    case linear:
        return "Linear";
        break;
    
    default:
        printf("Unexpected neuron type!\n");
        exit(1);
    }

    return nullptr;

}

Unit::Unit(size_t id, neuronTypes type)
{
    this->id_ = id;
    this->type_ = type;
}

Unit::~Unit()
{
}

const size_t Unit::Id()
{
    return this->id_;
}

const neuronTypes Unit::type()
{
    return this->type_;
}


nn_num_t Unit::Raw()
{
    return this->value_;
}
void Unit::ChangeConnectionWeight(Unit *from, nn_num_t delta)
{
    auto f = this->connections.find(Connection(from));
    if (f == this->connections.end())
    {
        printf("Error - couldn't find connection to change weight of!\n");
        exit(1);
    }
    Connection newC = *(f);
    this->connections.erase(f);
    newC.weight += delta;
    this->connections.insert(newC);
};

void Unit::SetConnectionWeight(Unit *from, nn_num_t w)
{
    auto f = this->connections.find(Connection(from));
    if (f == this->connections.end())
    {
        printf("Error - couldn't find connection to change weight of!\n");
        exit(1);
    }
    Connection newC = *(f);
    this->connections.erase(f);
    newC.weight = w;
    this->connections.insert(newC);
};

void Neuron::CalculateValue()
{
    this->value_ = 0.0;
    for (auto c : this->connections)
    {
        this->value_ += c.weight * c.from.u->Activation();
    }
}

// nn_num_t Unit::operator[](Unit *from)
// {
// return this->connectionWeights.find(Connection(from))->weight;
// };

const std::set<Connection> &Unit::GetConnections()
{
    return this->connections;
};

void Unit::AddConnection(Unit *u, nn_num_t w)
{
    this->connections.insert(Connection(u, w));
};

void Unit::RemoveConnection(Unit *u)
{
    this->connections.erase(Connection(u));
};

void Unit::Disconnect()
{
    this->connections.clear();
}

// Neuron

Neuron::Neuron(size_t id, neuronTypes type) : Unit(id, type)
{
}

Neuron::~Neuron()
{
}

// Input
Input::Input(size_t id) : Unit(id, input)
{
    this->value_ = 0.0;
}

nn_num_t Input::Activation()
{
    return this->value_;
}

nn_num_t Input::ActivationDeriv()
{
    return 0.0;
}

void Input::CalculateValue()
{
}

void Input::SetValue(nn_num_t newValue)
{
    this->value_ = newValue;
}

// Logistic
Logistic::Logistic(size_t id) : Neuron(id, logistic)
{
}

nn_num_t Logistic::Activation()
{
    return 1.0 / (1.0 + exp(-1.0 * this->value_));
}

nn_num_t Logistic::ActivationDeriv()
{

    nn_num_t a = this->Activation();
    return a * (1.0 - a);
}

// Perceptron
Perceptron::Perceptron(size_t id) : Neuron(id, perceptron)
{
}

nn_num_t Perceptron::Activation()
{
    return (this->value_ > 0) ? 1.0 : 0.0;
}

nn_num_t Perceptron::ActivationDeriv()
{
    nn_num_t a = 1.0 / (1.0 + exp(-1.0 * this->value_));
    return (a * (1.0 - a));
}

// Linear

Linear::Linear(size_t id) : Neuron(id, linear)
{
}

nn_num_t Linear::Activation()
{
    return this->value_;
}

nn_num_t Linear::ActivationDeriv()
{
    return 1.0;
}

// bias
BiasNeuron::BiasNeuron(size_t id) : Unit(id, bias)
{
    this->value_ = 1.0;
}

nn_num_t BiasNeuron::Activation()
{
    return this->value_;
}

nn_num_t BiasNeuron::ActivationDeriv()
{
    return 0.0;
}

void BiasNeuron::CalculateValue()
{
}
