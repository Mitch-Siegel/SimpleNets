#ifndef __CONNECTION_H_
#define __CONNECTION_H_
#include <math.h>

typedef float nn_num_t;

namespace SimpleNets
{

    class Unit;

    class Connection
    {
    private:
        bool idOnly = false;

    public:
        union
        {
            Unit *u;
            size_t id;
        } from;

        nn_num_t weight;

        Connection(Unit *u, nn_num_t weight);
        explicit Connection(Unit *u);
        explicit Connection(size_t id);

        bool operator==(const Connection &b);

        bool operator<(const Connection &b) const;
    };

} // namespace SimpleNets

#endif
