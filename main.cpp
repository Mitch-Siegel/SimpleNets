#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(1);
    n.AddLayer(1, NeuralNet::perceptron);
    n.ConfigureOutput(NeuralNet::output_singular, NeuralNet::perceptron);
    n.setInput(0);
    // n.dump();
    n.BackPropagate(0.0);
    n.UpdateWeights(1.0);
    printf("%f\n", n.Output());
    n.dump();
}