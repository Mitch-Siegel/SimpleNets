#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(2);
    // n.AddLayer(2, NeuralNet::linear);
    n.ConfigureOutput(1, NeuralNet::perceptron);
    n.Dump();
    printf("\n\n\n\n\n");

    for (int i = 0; i < 120; i++)
    {
        for (int a = 0; a < 2; a++)
        {
            for (int b = 0; b < 2; b++)
            {
                n.SetInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
                bool result = a & b;
                printf("Input %d,%d: output %f - expected %d - %s\n\n",
                       a, b,
                       n.Output(), a & b,
                       (static_cast<nn_num_t>(result) == n.Output()) ? "[PASS]" : "[FAIL]");
                n.Learn({static_cast<nn_num_t>(result)}, 1.0);
            }
        }
        if (i == 3)
        {
            printf("\n\nInserting new layer!\n");
            n.AddLayer(3, NeuralNet::linear);
            // n.dump();
            // exit(1);
        }
        else if(i == 6)
        {
            printf("\n\nRemoving from layer!\n");
            n.RemoveNeuron({1, 2});
        }
    }
    n.Dump();
    printf("\n\n\n\n");
    // n.AddLayer(1, NeuralNet::perceptron);

    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 2; b++)
        {
            n.SetInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
            printf("Input %d,%d: output %f - expected %d - %s\n",
                   a, b,
                   n.Output(), a & b,
                   (static_cast<nn_num_t>(a & b) == n.Output()) ? "[PASS]" : "[FAIL]");
        }
    }

    printf("\n\n");
    n.Dump();

    for (size_t i = 1; i < n.size(); i++)
    {
        for (size_t j = 0; j < n.size(i); j++)
        {
            for (size_t k = 0; k < n.size(i - 1); k++)
            {
                nn_num_t w_ij = n.GetWeight({i - 1, k}, {i, j});
                if (w_ij != 0.0)
                {
                    printf("%lu:%lu -> %lu:%lu: % 2.6f\n", i - 1, k, i, j, w_ij);
                }
            }
        }
    }

}