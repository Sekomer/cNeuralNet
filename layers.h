#ifndef __LAYERS__
#define __LAYERS__


#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>


#include "utils.h"


/* Forward Struct Decleartions */
typedef struct _neuron Neuron;
typedef struct _layer Layer;
typedef struct _neural_network NN;


/* Forward Function Decleartions */
void init_network(NN *, int, char *, int, ...);
void layer_init(Layer *, int, int, char *, float, float);
void name_the_layer(Layer *object, int, int);
void destroy_network(NN *);
void destroy_layer(Layer *);


struct _neuron {
    int size;

    float *weight_array;
    float  bias;

    float output;
    float delta;
};


struct _layer {
    char layer_name[16];
    char *activation_function;

    int number_of_neuron;
    int neuron_size;

    Neuron *neurons;

    //float *outputs;
    float *errors;

    bool is_input_layer;
    bool is_output_layer; 
};


struct _neural_network {
    char *network_name;
    int num_layers;

    Layer *layers;
};

#endif
