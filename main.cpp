#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(2);
    // n.AddLayer(2, NeuralNet::linear);
    n.ConfigureOutput(1, NeuralNet::perceptron);
    n.dump();
    printf("\n\n\n\n\n");

    for (int i = 0; i < 1000; i++)
    {
        for (int a = 0; a < 2; a++)
        {
            for (int b = 0; b < 2; b++)
            {
                n.setInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
                bool result = a & b;
                printf("Input %d,%d: output %f - expected %d - %s\n\n",
                       a, b,
                       n.Output(), a & b,
                       (static_cast<nn_num_t>(result) == n.Output()) ? "[PASS]" : "[FAIL]");
                n.Learn({static_cast<nn_num_t>(result)}, 1.0);
            }
        }
    }

    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 2; b++)
        {
            n.setInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
            printf("Input %d,%d: output %f - expected %d - %s\n",
                   a, b,
                   n.Output(), a & b,
                   (static_cast<nn_num_t>(a & b) == n.Output()) ? "[PASS]" : "[FAIL]");
        }
    }

    printf("\n\n");
    n.dump();
}