#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(1);
    n.AddLayer(1, NeuralNet::perceptron);
    n.ConfigureOutput(NeuralNet::output_singular, NeuralNet::perceptron);
    n.dump();
}