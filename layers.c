#include "layers.h"


void layer_init(Layer *object, int number_of_neuron, int neuron_size, char *activation, float initial_weight, float initial_bias)
{
    object->number_of_neuron = number_of_neuron;
    object->neuron_size = neuron_size;
    object->activation_function = activation;

    object->neurons = calloc(object->number_of_neuron, sizeof(Neuron));
    object->errors = calloc(object->number_of_neuron, sizeof(float));
    //object->outputs = calloc(number_of_neuron, sizeof(float));

    if (object->neurons == NULL)
    {
        fprintf(stderr, "Neuron allocation error! \n"); return;
    }

        
    for (int i = 0; i < object->number_of_neuron; ++i)
    {
        object->neurons[i].weight_array = calloc(object->neuron_size, sizeof(float));
        object->neurons[i].size = neuron_size;

        for (int j = 0; j < object->neuron_size; ++j)
            object->neurons[i].weight_array[j] = ((float)rand()/(float)(RAND_MAX))/10;

        object->neurons[i].bias = ((float)rand()/(float)(RAND_MAX))/10;
    }
}


void init_network(NN *model, int input_size, char *name, int num_layers, ...)
{
    model->num_layers = num_layers;

    va_list ap;
    va_start(ap, num_layers); 
    
    model->layers = calloc(num_layers, sizeof(Layer));

    for (int i = 0; i < num_layers; ++i)
    {
        int num_of_neuron = (va_arg(ap, int));
        int neuron_w_size;

        if (i == 0)
            neuron_w_size = input_size;
        else
            neuron_w_size = model->layers[i-1].number_of_neuron;

        // doesn't send initial weight and bias right now, they are random
        layer_init(&(model->layers[i]), num_of_neuron, neuron_w_size, "sig", -1, -1);
        name_the_layer(&(model->layers[i]), i, num_layers);
    }

    va_end(ap);
}

void name_the_layer(Layer *object, int layer_id, int num_layer)
{
    if (layer_id == 0)
    {	
        strcpy(object->layer_name, "input_layer");
    }
    else if (layer_id == num_layer - 1)
    {	
        strcpy(object->layer_name, "output_layer");
    }
    else
    {
        strcpy(object->layer_name, "hidden_layer_");
        object->layer_name[13] = (layer_id + 48);
        object->layer_name[14] = '\0';
    }
}


void destroy_network(NN *model)
{
    for (int i = 0; i < model->num_layers; ++i)
    {
        destroy_layer(&model->layers[i]);    
    }
    free(model->layers);
    model->layers = NULL;
}

void destroy_layer(Layer *target)
{
    for (int i = 0; i < target->number_of_neuron; ++i)
    {
        free(target->neurons[i].weight_array);
        target->neurons[i].weight_array = NULL;
    }
    free(target->neurons);
    target->neurons = NULL;
}

