#include <stdio.h>
#include "feedforwardnn.h"
#include "dagnn.h"

void testFeedForwardNet()
{
    SimpleNets::FeedForwardNeuralNet n(2, {{1, SimpleNets::perceptron}}, {1, SimpleNets::perceptron});

    bool needMoreTraining = true;
    size_t i = 0;
    while (needMoreTraining)
    {
        printf("Training epoch %lu\n", i++);
        needMoreTraining = false;
        for (int a = 0; a < 2; a++)
        {
            for (int b = 0; b < 2; b++)
            {
                n.SetInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
                bool result = a & b;
                printf("Input %d,%d: output %f - expected %d - %s\n",
                       a, b,
                       n.Output(), result,
                       (static_cast<nn_num_t>(result) == n.Output()) ? "[PASS]" : "[FAIL]");
                needMoreTraining |= static_cast<nn_num_t>(result) != n.Output();
                n.Learn({static_cast<nn_num_t>(result)}, 1.0);
            }
        }
    }
    printf("\n\nTesting:\n");

    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 2; b++)
        {
            n.SetInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
            bool result = a & b;

            printf("Input %d,%d: output %f - expected %d - %s\n",
                   a, b,
                   n.Output(), result,
                   (static_cast<nn_num_t>(result) == n.Output()) ? "[PASS]" : "[FAIL]");
        }
    }
    printf("\n\n");
    n.Dump();
}

void testDAGNet()
{
    SimpleNets::DAGNetwork n(2, {{SimpleNets::perceptron, 10}}, {1, SimpleNets::perceptron});
    n.Dump();
    n.AddConnection(0, 10, -0.0119899);
    n.AddConnection(1, 10, 0.0086526);
    n.AddConnection(2, 10, 0.0066443);
    n.AddConnection(10, 3, 0.0292656);
    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 2; b++)
        {
            n.SetInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
            bool result = a & b;

            printf("Input %d,%d: output %f - expected %d - %s\n",
                   a, b,
                   n.Output(), result,
                   (static_cast<nn_num_t>(result) == n.Output()) ? "[PASS]" : "[FAIL]");
        }
    }
    printf("\n\n");
    n.Dump();

}

int main()
{
    testFeedForwardNet();
    // testDAGNet();
    return 0;
}