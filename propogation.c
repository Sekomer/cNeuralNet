#include "layers.h"
#include <math.h>
#include "utils.h"
#include "propogation.h"

float sigmoid(float input)
{
    // math.h exp returns double
	return (1.0 / (1.0 + exp(-input)));
}

float sigmoid_derivative(float input)
{
	return input * (1.0 - input);
}


void train_model(NN *model, float **dataset, int dataset_size, float **labels, int epoch, bool verbose, char *optimisation, float alpha)
{
    for (int iteration = 0; iteration < epoch; ++iteration)
    {
        for (int row = 0; row < dataset_size; ++row)
        {    
            forward_prop(model->layers, dataset[row], model->num_layers);
            backward_prop(model->layers, labels[row], model->num_layers);
            update_weights(model->layers, dataset[row], model->num_layers, alpha);
            
            //logstr("outputs:");
            //logfloat(model->layers[model->num_layers - 1].neurons[0].output);
            //logfloat(model->layers[model->num_layers - 1].neurons[1].output);
            //logfloat(model->layers[model->num_layers - 1].neurons[2].output);
            //logfloat(model->layers[model->num_layers - 1].neurons[3].output);
        }
    }
}


void forward_prop(Layer *layers, float *input, int number_of_layers)
{
    for (int layer_id = 0; layer_id < number_of_layers; layer_id++)
    {
        for (int neuron_id = 0; neuron_id < layers[layer_id].number_of_neuron; ++neuron_id)
        {
            float sum = 0;
            if (layer_id == 0)    
            {    
                for (int i = 0; i < layers[layer_id].neuron_size; ++i)
                {
                    float activation = input[i] * layers[layer_id].neurons[neuron_id].weight_array[i] + layers[layer_id].neurons[neuron_id].bias;
                    sum += activation;
                }
                layers[layer_id].neurons[neuron_id].output = sigmoid(sum);
                
            }
            else
            {
                for (int i = 0; i < layers[layer_id].neuron_size; ++i)
                {
                    float activation = layers[layer_id-1].neurons[i].output * layers[layer_id].neurons[neuron_id].weight_array[i] + layers[layer_id].neurons[neuron_id].bias; 
                    sum += activation;
                }
                layers[layer_id].neurons[neuron_id].output = sigmoid(sum);
            }
        }
    }
}

void update_weights(Layer *layers, float *input, int number_of_layers, float learning_rate)
{

    for (int layer_id = 0; layer_id < number_of_layers; ++layer_id)
    {
        for (int neuron_id = 0; neuron_id < layers[layer_id].number_of_neuron; ++neuron_id)
        {
            for(int i = 0; i < layers[layer_id].neuron_size; ++i)
            {
                float delta = layers[layer_id].neurons[neuron_id].delta;

                // first layers previous is input 
                if (layer_id != 0)
                {    
                    /* logstr("prev");
                    logfloat(layers[layer_id].neurons[neuron_id].weight_array[i]); */
                    layers[layer_id].neurons[neuron_id].weight_array[i] += learning_rate * delta * layers[layer_id - 1].neurons[neuron_id].output;
                    /* logstr("later");
                    logfloat(layers[layer_id].neurons[neuron_id].weight_array[i]); */
                }
                else if (layer_id == 0)
                { 
                    layers[layer_id].neurons[neuron_id].weight_array[i] += learning_rate * delta * input[neuron_id];
                }
            }
            // bias
            layers[layer_id].neurons[neuron_id].bias += learning_rate * layers[layer_id].neurons[neuron_id].delta; 
        }
    }
}

void backward_prop(Layer *layers, float *labels, int number_of_layers)
{
    for (int layer_id = number_of_layers - 1; 0 <= layer_id; --layer_id)
    {
        if (layer_id != number_of_layers - 1)
        {
            for (int neuron_id = 0; neuron_id < layers[layer_id].number_of_neuron; ++neuron_id)
            {
                float error  = 0;
                
                // next layer
                for (int next = 0; next < layers[layer_id + 1].number_of_neuron; ++next)
                {
                    error += layers[layer_id + 1].neurons[next].weight_array[neuron_id] * layers[layer_id + 1].neurons[next].delta;
                }
                layers[layer_id].errors[neuron_id] = error;
            }
        }
        
        /* Label Layer */
        else if (layer_id == number_of_layers - 1)
        {
            for (int neuron_id = 0; neuron_id < layers[layer_id].number_of_neuron; ++neuron_id)
            {
                float error = labels[neuron_id] - layers[layer_id].neurons[neuron_id].output;
                layers[layer_id].errors[neuron_id] = error;     
                //logfloat(error);     
            }
        }
        for (int neuron_id = 0; neuron_id < layers[layer_id].number_of_neuron; neuron_id++)
        {
            float delta = layers[layer_id].errors[neuron_id] * sigmoid_derivative(layers[layer_id].neurons[neuron_id].output);

            layers[layer_id].neurons[neuron_id].delta = delta;
        }
    }

}



void matrix_alloc(struct Matrix *obj)
{
    // usage //
    /* array[x][y] */

	/* size_x is the width of the array */
	float **array = calloc(obj->size_x, sizeof(float*));

	for(int i = 0; i < obj->size_x; i++) {
        /* size_y is the height */
    	array[i] = calloc(obj->size_y, sizeof(float));
	}

	obj->array = array;
}

void matrix_randomize(struct Matrix *obj)
{
    for(int i = 0; i < obj->size_x; ++i)
        for(int j = 0; j < obj->size_y; ++j)
        {
            obj->array[i][j] =  ((float) rand() / (float) RAND_MAX);
        }
}

void matrix_free(struct Matrix *obj)
{
    for(int i = 0; i < obj->size_x; ++i) {
        free(obj->array[i]);
    }

    free(obj->array);
}