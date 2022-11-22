#include <stdio.h>
#include "nnet.h"

int main()
{
    printf("Hello, world!\n");
    NeuralNet n(1);
    n.AddLayer(1, NeuralNet::perceptron);
    n.ConfigureOutput(NeuralNet::output_singular, NeuralNet::perceptron);
    n.setInput(0);
    for (int i = 1; i < 10; i++)
    {
        printf("%d expects %d\n", (i + 1) % 2, i % 2);
        n.setInput(((i + 1) % 2) + 1);
        n.ForwardPropagate();
        n.BackPropagate(i % 2);
        n.UpdateWeights(1.0 / i);
    }
    printf("%f\n", n.Output());
    n.dump();
    printf("\n\n\n");
    n.ForwardPropagate();
    printf("%f\n", n.Output());
    n.dump();

    n.setInput(0);
    n.ForwardPropagate();
    printf("0:%f\n", n.Output());

    n.setInput(1);
    n.ForwardPropagate();
    printf("1:%f\n", n.Output());
}