#include "connection.h"
#include "units.h"

namespace SimpleNets
{
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
}