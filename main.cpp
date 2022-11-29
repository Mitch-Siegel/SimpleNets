#include <stdio.h>
#include "feedforwardnn.h"

int main()
{
    printf("Hello, world!\n");
    SimpleNets::FeedForwardNeuralNet n(2, {}, {1, SimpleNets::perceptron});
    printf("\n\n");

    for (int i = 0; i < 5; i++)
    {
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
                n.Learn({static_cast<nn_num_t>(result)}, 1.0);
            }
        }
    }
    printf("\n\nTesting:\n");
    // n.AddLayer(1, NeuralNet::perceptron);

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