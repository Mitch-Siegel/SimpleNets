#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(1);
    n.AddLayer(1, NeuralNet::logistic);
    n.ConfigureOutput(NeuralNet::output_singular, NeuralNet::linear);
    n.dump();
    printf("\n\n\n\n\n");

    for (int i = 2; i < 100; i++)
    {
        n.setInput(i % 2);
        n.ForwardPropagate();
        printf("Input %d: output %f - expected %d\n\n", i % 2, n.Output(), (i + 1) % 2);
        n.BackPropagate((i + 1) % 2);
        // n.UpdateWeights(1.0 / (nn_num_t)sqrt(i / 2));
        n.UpdateWeights(1.0);
        // n.dump();
        // printf("\n\n\n\n\n");
    }
    n.setInput(0);
    n.ForwardPropagate();
    printf("input 0: %f\n", n.Output());
    n.setInput(1);
    n.ForwardPropagate();
    printf("input 1: %f\n", n.Output());
    printf("\n\n");
    n.dump();

}