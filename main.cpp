#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(1);
    n.AddLayer(1, NeuralNet::logistic);

    n.ConfigureOutput(2, NeuralNet::linear);
    n.dump();
    printf("\n\n\n\n\n");

    for (int i = 2; i < 100; i++)
    {
        n.setInput({(nn_num_t)(i % 2)});
        n.ForwardPropagate();
        printf("Input %f: output %f - expected %d\n\n", (nn_num_t)(i % 2), n.Output(), (i + 1) % 2);
        n.BackPropagate({(nn_num_t)((i + 0) % 2), (nn_num_t)((i + 1) % 2)});
        // n.UpdateWeights(1.0 / (nn_num_t)sqrt(i / 2));
        n.UpdateWeights(1.0);
        // n.dump();
        // printf("\n\n\n\n\n");
    }
    n.setInput({0});
    n.ForwardPropagate();
    printf("input 0.0: %f\n", n.Output());
    n.setInput({1});
    n.ForwardPropagate();
    printf("input 1.0: %f\n", n.Output());
    printf("\n\n");
    n.dump();

}