#include <stdlib.h>

float* random_array(int size)
{
    float *test = calloc(size, sizeof(float));
    for (int i = 0; i < size; ++i)
    {
        test[i] = ((float)rand()/(float)(RAND_MAX));
    }
    return test;
}