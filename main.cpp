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
    n.AddConnection(0, 10, 0.1);
    n.AddConnection(1, 10, 0.1);
    n.AddConnection(2, 10, 0.1);
    n.AddConnection(10, 3, 0.1);
    n.Dump();
    
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
                nn_num_t result = static_cast<nn_num_t>(a & b);
                nn_num_t output = n.Output();
                printf("Input %d,%d: output %lf - expected %lf - %s\n",
                       a, b,
                       output, result,
                       (result == output) ? "[PASS]" : "[FAIL]");
                needMoreTraining |= (result != output);
                n.Learn({result}, 1.0);
                n.Dump();
            }
        }
        if (needMoreTraining)
        {
            printf("need more training...\n");
        }
        if(i > 10)
        {
            printf("Hit epoch limit!\n");
            break;
        }
    }
    printf("\n\nTesting:\n");

    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 2; b++)
        {
            n.SetInput({static_cast<nn_num_t>(a), static_cast<nn_num_t>(b)});
            nn_num_t result = static_cast<nn_num_t>(a & b);
            nn_num_t output = n.Output();
            printf("Input %d,%d: output %f - expected %f - %s\n",
                   a, b,
                   output, result,
                   (result == output) ? "[PASS]" : "[FAIL]");
        }
    }
    printf("\n\n");
    n.Dump();
}

int main()
{
    // testFeedForwardNet();
    testDAGNet();
    return 0;
}