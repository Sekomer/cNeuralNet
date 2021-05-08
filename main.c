#include "layers.h"
#include "propogation.h"

#define INPUT_SIZE      16
#define NUM_LAYERS      3

#define OUT(network)    for(int i = 0; i < network.layers[NUM_LAYERS - 1].number_of_neuron ; ++i) {     \
                            logfloat(network.layers[NUM_LAYERS - 1].neurons[i].output);                 \
                        }                                                                               \

int main()
{
    srand(31);

    Matrix dataset = {.size_x = 64, .size_y = 16};
    Matrix labels  = {.size_x = 64, .size_y = 4};

    matrix_alloc(&dataset);
    matrix_alloc(&labels);
    
    matrix_randomize(&dataset);
    matrix_randomize(&labels);

    /* label for testing */
    /* [ 0 0 1 0 ] */
    for(int i = 0; i < 64; ++i)
    {
        labels.array[i][0] = 0;
        labels.array[i][1] = 0;
        labels.array[i][2] = 1;
        labels.array[i][3] = 0;
    }

    NN network;
    init_network(&network, INPUT_SIZE, "Test Network", NUM_LAYERS, 4, 8, 4);
    train_model(&network, dataset.array, 64, labels.array, 1000, 1, "sigmoid", .1);

    logstr("trained"); 
    OUT(network);
    /* 
        trained
        0.002552 [ ]
        0.002500 [ ]
        0.997493 [*]
        0.002515 [ ]
    */
    
    
    destroy_network(&network);
    matrix_free(&dataset);
    matrix_free(&labels);

return 0;
}

