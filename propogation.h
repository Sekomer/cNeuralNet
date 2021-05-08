#ifndef __PROPOGATION__
#define __PROPOGATION__

typedef struct Matrix Matrix;

struct Matrix {
    float **array;
    int size_x;
    int size_y;
};


void matrix_alloc(struct Matrix *obj);
void matrix_free(struct Matrix *obj);
void matrix_randomize(struct Matrix *obj);

void train_model(NN *model, float **dataset, int dataset_size, float **labels, int epoch, bool verbose, char *optimisation, float alpha);
void forward_prop(Layer *layers, float *input, int number_of_layers);
void backward_prop(Layer *layers, float *labels, int number_of_layers);
void update_weights(Layer *layers, float *input, int number_of_layers, float learning_rate);
float sigmoid(float);
float sigmoid_derivative(float);



#endif