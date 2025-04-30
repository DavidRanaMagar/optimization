/**
 * Animal Image Classifier CNN
 *
 * This program implements a Convolutional Neural Network in C
 * for classifying images of animals (cats, dogs, snakes).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "stb_image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define IMAGE_CHANNELS 3
#define NUM_CLASSES 3
#define BATCH_SIZE 32
#define LEARNING_RATE 0.001
#define NUM_EPOCHS 10


#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif


// Class labels
const char* CLASS_NAMES[NUM_CLASSES] = {"cats", "dogs", "snakes"};

// CNN Layer Parameters
typedef struct {
    int width;
    int height;
    int depth;
    float* data;
} Tensor;

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    float* weights;
    float* bias;
    float* d_weights;
    float* d_bias;
} ConvLayer;

typedef struct {
    int size;
    int* max_indices;  // Store indices of max values for backprop
} MaxPoolLayer;

typedef struct {
    int input_size;
    int output_size;
    float* weights;
    float* bias;
    float* d_weights;
    float* d_bias;
} FCLayer;

// CNN Architecture
typedef struct {
    Tensor* input;

    ConvLayer* conv1;
    Tensor* conv1_output;

    MaxPoolLayer* pool1;
    Tensor* pool1_output;

    ConvLayer* conv2;
    Tensor* conv2_output;

    MaxPoolLayer* pool2;
    Tensor* pool2_output;

    FCLayer* fc1;
    float* fc1_output;
    float* fc1_preactivation;  // Store pre-activation values for backprop

    FCLayer* fc2;
    float* fc2_output;
    float* fc2_preactivation;  // Store pre-activation values for backprop

    float* softmax_output;

    // For backpropagation
    float* gradients;
} CNN;

// Helper functions
float random_float() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

// Memory allocation functions
Tensor* create_tensor(int width, int height, int depth) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->width = width;
    tensor->height = height;
    tensor->depth = depth;
    tensor->data = (float*)calloc(width * height * depth, sizeof(float));
    return tensor;
}

void free_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor);
}

ConvLayer* create_conv_layer(int in_channels, int out_channels, int kernel_size) {
    ConvLayer* layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;

    int weights_size = out_channels * in_channels * kernel_size * kernel_size;
    layer->weights = (float*)malloc(weights_size * sizeof(float));
    layer->d_weights = (float*)calloc(weights_size, sizeof(float));

    layer->bias = (float*)malloc(out_channels * sizeof(float));
    layer->d_bias = (float*)calloc(out_channels, sizeof(float));

    // Initialize weights with Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (in_channels * kernel_size * kernel_size + out_channels));
    for (int i = 0; i < weights_size; i++) {
        layer->weights[i] = random_float() * scale;
    }

    // Initialize bias to zero
    for (int i = 0; i < out_channels; i++) {
        layer->bias[i] = 0.0f;
    }

    return layer;
}

void free_conv_layer(ConvLayer* layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer->d_weights);
    free(layer->d_bias);
    free(layer);
}

MaxPoolLayer* create_max_pool_layer(int size, int input_width, int input_height, int input_depth) {
    MaxPoolLayer* layer = (MaxPoolLayer*)malloc(sizeof(MaxPoolLayer));
    layer->size = size;

    // Allocate memory for max indices (for backpropagation)
    int output_width = input_width / size;
    int output_height = input_height / size;
    layer->max_indices = (int*)malloc(output_width * output_height * input_depth * sizeof(int));

    return layer;
}

void free_max_pool_layer(MaxPoolLayer* layer) {
    free(layer->max_indices);
    free(layer);
}

FCLayer* create_fc_layer(int input_size, int output_size) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->weights = (float*)malloc(input_size * output_size * sizeof(float));
    layer->d_weights = (float*)calloc(input_size * output_size, sizeof(float));

    layer->bias = (float*)malloc(output_size * sizeof(float));
    layer->d_bias = (float*)calloc(output_size, sizeof(float));

    // Initialize weights with Xavier/Glorot initialization
    float scale = sqrtf(2.0f / (input_size + output_size));
    for (int i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = random_float() * scale;
    }

    // Initialize bias to zero
    for (int i = 0; i < output_size; i++) {
        layer->bias[i] = 0.0f;
    }

    return layer;
}

void free_fc_layer(FCLayer* layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer->d_weights);
    free(layer->d_bias);
    free(layer);
}

// CNN Operations
void conv_forward(ConvLayer* layer, Tensor* input, Tensor* output) {
    int stride = 1;
    int pad = layer->kernel_size / 2;

    // Zero the output tensor
    memset(output->data, 0, output->width * output->height * output->depth * sizeof(float));

    // For each output channel
    for (int out_c = 0; out_c < layer->out_channels; out_c++) {
        // For each spatial position in the output
        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                float sum = layer->bias[out_c];

                // For each input channel
                for (int in_c = 0; in_c < layer->in_channels; in_c++) {
                    // For each kernel position
                    for (int ky = 0; ky < layer->kernel_size; ky++) {
                        for (int kx = 0; kx < layer->kernel_size; kx++) {
                            int input_y = y * stride + ky - pad;
                            int input_x = x * stride + kx - pad;

                            // Check if we're inside the input bounds
                            if (input_y >= 0 && input_y < input->height &&
                                input_x >= 0 && input_x < input->width) {

                                float input_val = input->data[(in_c * input->height + input_y) * input->width + input_x];
                                int w_idx = ((out_c * layer->in_channels + in_c) * layer->kernel_size + ky) * layer->kernel_size + kx;

                                sum += input_val * layer->weights[w_idx];
                            }
                        }
                    }
                }

                // Apply ReLU activation
                output->data[(out_c * output->height + y) * output->width + x] = relu(sum);
            }
        }
    }
}

void conv_backward(ConvLayer* layer, Tensor* input, Tensor* output, Tensor* d_output, Tensor* d_input, float learning_rate) {
    int stride = 1;
    int pad = layer->kernel_size / 2;

    // Initialize d_input to zero
    if (d_input != NULL) {
        memset(d_input->data, 0, d_input->width * d_input->height * d_input->depth * sizeof(float));
    }

    // For each output channel
    for (int out_c = 0; out_c < layer->out_channels; out_c++) {
        // For each spatial position in the output
        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                int out_idx = (out_c * output->height + y) * output->width + x;
                float d_output_val = d_output->data[out_idx];

                // Only backpropagate through the activated neurons (ReLU derivative)
                if (output->data[out_idx] > 0) {
                    // Update bias
                    layer->d_bias[out_c] += d_output_val;

                    // For each input channel
                    for (int in_c = 0; in_c < layer->in_channels; in_c++) {
                        // For each kernel position
                        for (int ky = 0; ky < layer->kernel_size; ky++) {
                            for (int kx = 0; kx < layer->kernel_size; kx++) {
                                int input_y = y * stride + ky - pad;
                                int input_x = x * stride + kx - pad;

                                // Check if we're inside the input bounds
                                if (input_y >= 0 && input_y < input->height &&
                                    input_x >= 0 && input_x < input->width) {

                                    int input_idx = (in_c * input->height + input_y) * input->width + input_x;
                                    float input_val = input->data[input_idx];

                                    // Compute weight gradient
                                    int w_idx = ((out_c * layer->in_channels + in_c) * layer->kernel_size + ky) * layer->kernel_size + kx;
                                    layer->d_weights[w_idx] += input_val * d_output_val;

                                    // Compute input gradient (if needed)
                                    if (d_input != NULL) {
                                        d_input->data[input_idx] += layer->weights[w_idx] * d_output_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Update weights and biases
    int weights_size = layer->out_channels * layer->in_channels * layer->kernel_size * layer->kernel_size;
    for (int i = 0; i < weights_size; i++) {
        layer->weights[i] -= learning_rate * layer->d_weights[i];
        layer->d_weights[i] = 0; // Reset gradients
    }

    for (int i = 0; i < layer->out_channels; i++) {
        layer->bias[i] -= learning_rate * layer->d_bias[i];
        layer->d_bias[i] = 0; // Reset gradients
    }
}

void max_pool_forward(MaxPoolLayer* layer, Tensor* input, Tensor* output) {
    int stride = layer->size;

    // For each output channel
    for (int c = 0; c < output->depth; c++) {
        // For each spatial position in the output
        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                float max_val = -INFINITY;
                int max_idx = -1;

                // Find maximum in the pool region
                for (int py = 0; py < layer->size; py++) {
                    for (int px = 0; px < layer->size; px++) {
                        int input_y = y * stride + py;
                        int input_x = x * stride + px;

                        if (input_y < input->height && input_x < input->width) {
                            int input_idx = (c * input->height + input_y) * input->width + input_x;
                            float val = input->data[input_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = input_idx;
                            }
                        }
                    }
                }

                int output_idx = (c * output->height + y) * output->width + x;
                output->data[output_idx] = max_val;
                layer->max_indices[output_idx] = max_idx;  // Store max index for backprop
            }
        }
    }
}

void max_pool_backward(MaxPoolLayer* layer, Tensor* input, Tensor* output, Tensor* d_output, Tensor* d_input) {
    // Initialize d_input to zero
    memset(d_input->data, 0, d_input->width * d_input->height * d_input->depth * sizeof(float));

    // For each output position
    for (int c = 0; c < output->depth; c++) {
        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                int output_idx = (c * output->height + y) * output->width + x;
                int max_idx = layer->max_indices[output_idx];

                // Propagate gradient only to the max element
                if (max_idx >= 0) {
                    d_input->data[max_idx] += d_output->data[output_idx];
                }
            }
        }
    }
}

void fc_forward(FCLayer* layer, float* input, float* output, float* preactivation) {
    // For each output neuron
    for (int i = 0; i < layer->output_size; i++) {
        float sum = layer->bias[i];

        // For each input connection
        for (int j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[i * layer->input_size + j];
        }

        // Store pre-activation
        preactivation[i] = sum;

        // Apply ReLU activation
        output[i] = relu(sum);
    }
}

void fc_backward(FCLayer* layer, float* input, float* output, float* preactivation, float* d_output, float* d_input, float learning_rate) {
    // Initialize d_input to zero if needed
    if (d_input != NULL) {
        memset(d_input, 0, layer->input_size * sizeof(float));
    }

    // For each output neuron
    for (int i = 0; i < layer->output_size; i++) {
        // Apply ReLU derivative
        float d_preactivation = d_output[i] * relu_derivative(preactivation[i]);

        // Update bias
        layer->d_bias[i] += d_preactivation;

        // For each input connection
        for (int j = 0; j < layer->input_size; j++) {
            // Update weights
            layer->d_weights[i * layer->input_size + j] += input[j] * d_preactivation;

            // Compute gradients for the previous layer
            if (d_input != NULL) {
                d_input[j] += layer->weights[i * layer->input_size + j] * d_preactivation;
            }
        }
    }

    // Update weights and biases
    for (int i = 0; i < layer->output_size * layer->input_size; i++) {
        layer->weights[i] -= learning_rate * layer->d_weights[i];
        layer->d_weights[i] = 0; // Reset gradients
    }

    for (int i = 0; i < layer->output_size; i++) {
        layer->bias[i] -= learning_rate * layer->d_bias[i];
        layer->d_bias[i] = 0; // Reset gradients
    }
}

void softmax(float* input, float* output, int size) {
    float max_val = -INFINITY;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

float cross_entropy_loss(float* predictions, int target_class, int num_classes) {
    float loss = -logf(fmaxf(predictions[target_class], 1e-7));
    return loss;
}

void softmax_cross_entropy_backward(float* predictions, int target_class, int num_classes, float* gradients) {
    // Gradients for softmax cross-entropy loss: p_i - 1(i == y)
    for (int i = 0; i < num_classes; i++) {
        gradients[i] = predictions[i];
    }
    gradients[target_class] -= 1.0f;
}

// CNN Creation and Forward Pass
CNN* create_cnn() {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));

    // Define the CNN architecture
    cnn->input = create_tensor(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS);

    // First convolutional layer: 3 input channels, 16 output channels, 3x3 kernel
    cnn->conv1 = create_conv_layer(IMAGE_CHANNELS, 16, 3);
    cnn->conv1_output = create_tensor(IMAGE_WIDTH, IMAGE_HEIGHT, 16);

    // First max pooling layer: 2x2 pooling
    cnn->pool1 = create_max_pool_layer(2, IMAGE_WIDTH, IMAGE_HEIGHT, 16);
    cnn->pool1_output = create_tensor(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, 16);

    // Second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
    cnn->conv2 = create_conv_layer(16, 32, 3);
    cnn->conv2_output = create_tensor(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, 32);

    // Second max pooling layer: 2x2 pooling
    cnn->pool2 = create_max_pool_layer(2, IMAGE_WIDTH/2, IMAGE_HEIGHT/2, 32);
    cnn->pool2_output = create_tensor(IMAGE_WIDTH/4, IMAGE_HEIGHT/4, 32);

    // Calculate the size of the flattened tensor after the second pooling layer
    int flattened_size = (IMAGE_WIDTH/4) * (IMAGE_HEIGHT/4) * 32;

    // First fully connected layer
    cnn->fc1 = create_fc_layer(flattened_size, 128);
    cnn->fc1_output = (float*)malloc(128 * sizeof(float));
    cnn->fc1_preactivation = (float*)malloc(128 * sizeof(float));

    // Output fully connected layer
    cnn->fc2 = create_fc_layer(128, NUM_CLASSES);
    cnn->fc2_output = (float*)malloc(NUM_CLASSES * sizeof(float));
    cnn->fc2_preactivation = (float*)malloc(NUM_CLASSES * sizeof(float));

    // Softmax output
    cnn->softmax_output = (float*)malloc(NUM_CLASSES * sizeof(float));

    // Gradients for backpropagation
    cnn->gradients = (float*)malloc(NUM_CLASSES * sizeof(float));

    return cnn;
}

void free_cnn(CNN* cnn) {
    free_tensor(cnn->input);

    free_conv_layer(cnn->conv1);
    free_tensor(cnn->conv1_output);

    free_max_pool_layer(cnn->pool1);
    free_tensor(cnn->pool1_output);

    free_conv_layer(cnn->conv2);
    free_tensor(cnn->conv2_output);

    free_max_pool_layer(cnn->pool2);
    free_tensor(cnn->pool2_output);

    free_fc_layer(cnn->fc1);
    free(cnn->fc1_output);
    free(cnn->fc1_preactivation);

    free_fc_layer(cnn->fc2);
    free(cnn->fc2_output);
    free(cnn->fc2_preactivation);

    free(cnn->softmax_output);
    free(cnn->gradients);

    free(cnn);
}

void cnn_forward(CNN* cnn) {
    // Forward pass through the CNN
    conv_forward(cnn->conv1, cnn->input, cnn->conv1_output);
    max_pool_forward(cnn->pool1, cnn->conv1_output, cnn->pool1_output);

    conv_forward(cnn->conv2, cnn->pool1_output, cnn->conv2_output);
    max_pool_forward(cnn->pool2, cnn->conv2_output, cnn->pool2_output);

    // Flatten the output from the last pooling layer
    int flattened_size = cnn->pool2_output->width * cnn->pool2_output->height * cnn->pool2_output->depth;
    float* flattened = (float*)malloc(flattened_size * sizeof(float));

    for (int c = 0; c < cnn->pool2_output->depth; c++) {
        for (int y = 0; y < cnn->pool2_output->height; y++) {
            for (int x = 0; x < cnn->pool2_output->width; x++) {
                int idx = (c * cnn->pool2_output->height + y) * cnn->pool2_output->width + x;
                flattened[idx] = cnn->pool2_output->data[idx];
            }
        }
    }

    // Forward through fully connected layers
    fc_forward(cnn->fc1, flattened, cnn->fc1_output, cnn->fc1_preactivation);
    fc_forward(cnn->fc2, cnn->fc1_output, cnn->fc2_output, cnn->fc2_preactivation);

    // Apply softmax to get class probabilities
    softmax(cnn->fc2_output, cnn->softmax_output, NUM_CLASSES);

    free(flattened);
}

void cnn_backward(CNN* cnn, int target_class, float learning_rate) {
    int flattened_size = cnn->pool2_output->width * cnn->pool2_output->height * cnn->pool2_output->depth;

    // Compute gradients for softmax cross-entropy loss
    softmax_cross_entropy_backward(cnn->softmax_output, target_class, NUM_CLASSES, cnn->gradients);

    // Backpropagation through fully connected layers
    float* d_fc1_output = (float*)malloc(128 * sizeof(float));
    fc_backward(cnn->fc2, cnn->fc1_output, cnn->fc2_output, cnn->fc2_preactivation, cnn->gradients, d_fc1_output, learning_rate);

    float* d_pool2_flattened = (float*)malloc(flattened_size * sizeof(float));
    fc_backward(cnn->fc1, d_pool2_flattened, cnn->fc1_output, cnn->fc1_preactivation, d_fc1_output, d_pool2_flattened, learning_rate);

    // Reshape flattened gradients back to pool2_output shape
    Tensor* d_pool2_output = create_tensor(cnn->pool2_output->width, cnn->pool2_output->height, cnn->pool2_output->depth);
    for (int c = 0; c < cnn->pool2_output->depth; c++) {
        for (int y = 0; y < cnn->pool2_output->height; y++) {
            for (int x = 0; x < cnn->pool2_output->width; x++) {
                int idx = (c * cnn->pool2_output->height + y) * cnn->pool2_output->width + x;
                d_pool2_output->data[idx] = d_pool2_flattened[idx];
            }
        }
    }

    // Backpropagation through convolutional and pooling layers
    Tensor* d_conv2_output = create_tensor(cnn->conv2_output->width, cnn->conv2_output->height, cnn->conv2_output->depth);
    max_pool_backward(cnn->pool2, cnn->conv2_output, cnn->pool2_output, d_pool2_output, d_conv2_output);

    Tensor* d_pool1_output = create_tensor(cnn->pool1_output->width, cnn->pool1_output->height, cnn->pool1_output->depth);
    conv_backward(cnn->conv2, cnn->pool1_output, cnn->conv2_output, d_conv2_output, d_pool1_output, learning_rate);

    Tensor* d_conv1_output = create_tensor(cnn->conv1_output->width, cnn->conv1_output->height, cnn->conv1_output->depth);
    max_pool_backward(cnn->pool1, cnn->conv1_output, cnn->pool1_output, d_pool1_output, d_conv1_output);

    // For the input layer, we don't need to compute gradients since there are no more layers
    conv_backward(cnn->conv1, cnn->input, cnn->conv1_output, d_conv1_output, NULL, learning_rate);

    // Clean up
    free(d_fc1_output);
    free(d_pool2_flattened);
    free_tensor(d_pool2_output);
    free_tensor(d_conv2_output);
    free_tensor(d_pool1_output);
    free_tensor(d_conv1_output);
}

// Data loading functions
void load_image(const char* filename, Tensor* tensor) {
    int width, height, channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);  // Force 3 channels

    if (!data) {
        printf("Failed to load image: %s\n", filename);
        return;
    }

    // Resize and normalize to [0,1]
    for (int c = 0; c < tensor->depth; c++) {
        for (int y = 0; y < tensor->height; y++) {
            for (int x = 0; x < tensor->width; x++) {
                // Simple nearest-neighbor resize
                int src_y = y * height / tensor->height;
                int src_x = x * width / tensor->width;
                int src_idx = (src_y * width + src_x) * 3;  // 3 channels RGB

                float val = data[src_idx + c] / 255.0f;
                tensor->data[(c * tensor->height + y) * tensor->width + x] = val;
            }
        }
    }

    stbi_image_free(data);
}

int parse_label(const char* path) {
    // Extract class from path
    if (strstr(path, "cats") != NULL) return 0;
    if (strstr(path, "dogs") != NULL) return 1;
    if (strstr(path, "snakes") != NULL) return 2;
    return -1; // Unknown class
}

// Training function
void train_cnn(CNN* cnn, const char** image_paths, int num_images) {
    printf("Starting training for %d epochs...\n", NUM_EPOCHS);

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;

        // Shuffle the data
        int* indices = (int*)malloc(num_images * sizeof(int));
        for (int i = 0; i < num_images; i++) {
            indices[i] = i;
        }

        // Fisher-Yates shuffle
        for (int i = num_images - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Process in batches
        for (int batch_start = 0; batch_start < num_images; batch_start += BATCH_SIZE) {
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > num_images) batch_end = num_images;

            for (int i = batch_start; i < batch_end; i++) {
                int idx = indices[i];

                // Load image
                load_image(image_paths[idx], cnn->input);

                // Get ground truth label
                int label = parse_label(image_paths[idx]);
                if (label < 0) continue;  // Skip if label can't be determined

                // Forward pass
                cnn_forward(cnn);

                // Calculate loss
                float loss = cross_entropy_loss(cnn->softmax_output, label, NUM_CLASSES);
                total_loss += loss;

                // Calculate accuracy
                int predicted_class = 0;
                float max_prob = cnn->softmax_output[0];
                for (int c = 1; c < NUM_CLASSES; c++) {
                    if (cnn->softmax_output[c] > max_prob) {
                        max_prob = cnn->softmax_output[c];
                        predicted_class = c;
                    }
                }

                if (predicted_class == label) {
                    correct++;
                }

                // Backward pass and update weights
                cnn_backward(cnn, label, LEARNING_RATE);
            }

            // Print progress every few batches
            if ((batch_start / BATCH_SIZE) % 5 == 0) {
                printf("Epoch %d/%d, Batch %d/%d\n",
                       epoch + 1, NUM_EPOCHS,
                       batch_start / BATCH_SIZE + 1,
                       (num_images + BATCH_SIZE - 1) / BATCH_SIZE);
            }
        }

        // Calculate and print epoch statistics
        float avg_loss = total_loss / num_images;
        float accuracy = (float)correct / num_images;
        printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1, NUM_EPOCHS, avg_loss, accuracy * 100.0f);

        free(indices);
    }

    printf("Training completed.\n");
}

// Prediction function
int predict(CNN* cnn, const char* image_path) {
    // Load image
    load_image(image_path, cnn->input);

    // Forward pass
    cnn_forward(cnn);

    // Find class with highest probability
    int predicted_class = 0;
    float max_prob = cnn->softmax_output[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (cnn->softmax_output[c] > max_prob) {
            max_prob = cnn->softmax_output[c];
            predicted_class = c;
        }
    }

    return predicted_class;
}

// Model saving/loading functions
void save_model(CNN* cnn, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file for writing: %s\n", filename);
        return;
    }

    // Save conv1 weights and biases
    int conv1_weights_size = cnn->conv1->out_channels * cnn->conv1->in_channels *
                             cnn->conv1->kernel_size * cnn->conv1->kernel_size;
    fwrite(cnn->conv1->weights, sizeof(float), conv1_weights_size, f);
    fwrite(cnn->conv1->bias, sizeof(float), cnn->conv1->out_channels, f);

    // Save conv2 weights and biases
    int conv2_weights_size = cnn->conv2->out_channels * cnn->conv2->in_channels *
                             cnn->conv2->kernel_size * cnn->conv2->kernel_size;
    fwrite(cnn->conv2->weights, sizeof(float), conv2_weights_size, f);
    fwrite(cnn->conv2->bias, sizeof(float), cnn->conv2->out_channels, f);

    // Save fc1 weights and biases
    int fc1_weights_size = cnn->fc1->input_size * cnn->fc1->output_size;
    fwrite(cnn->fc1->weights, sizeof(float), fc1_weights_size, f);
    fwrite(cnn->fc1->bias, sizeof(float), cnn->fc1->output_size, f);

    // Save fc2 weights and biases
    int fc2_weights_size = cnn->fc2->input_size * cnn->fc2->output_size;
    fwrite(cnn->fc2->weights, sizeof(float), fc2_weights_size, f);
    fwrite(cnn->fc2->bias, sizeof(float), cnn->fc2->output_size, f);

    fclose(f);
    printf("Model saved to %s\n", filename);
}

int load_model(CNN* cnn, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Could not open file for reading: %s\n", filename);
        return 0;
    }

    // Load conv1 weights and biases
    int conv1_weights_size = cnn->conv1->out_channels * cnn->conv1->in_channels *
                             cnn->conv1->kernel_size * cnn->conv1->kernel_size;
    size_t read = fread(cnn->conv1->weights, sizeof(float), conv1_weights_size, f);
    if (read != conv1_weights_size) {
        printf("Error reading conv1 weights\n");
        fclose(f);
        return 0;
    }

    read = fread(cnn->conv1->bias, sizeof(float), cnn->conv1->out_channels, f);
    if (read != cnn->conv1->out_channels) {
        printf("Error reading conv1 biases\n");
        fclose(f);
        return 0;
    }

    // Load conv2 weights and biases
    int conv2_weights_size = cnn->conv2->out_channels * cnn->conv2->in_channels *
                             cnn->conv2->kernel_size * cnn->conv2->kernel_size;
    read = fread(cnn->conv2->weights, sizeof(float), conv2_weights_size, f);
    if (read != conv2_weights_size) {
        printf("Error reading conv2 weights\n");
        fclose(f);
        return 0;
    }

    read = fread(cnn->conv2->bias, sizeof(float), cnn->conv2->out_channels, f);
    if (read != cnn->conv2->out_channels) {
        printf("Error reading conv2 biases\n");
        fclose(f);
        return 0;
    }

    // Load fc1 weights and biases
    int fc1_weights_size = cnn->fc1->input_size * cnn->fc1->output_size;
    read = fread(cnn->fc1->weights, sizeof(float), fc1_weights_size, f);
    if (read != fc1_weights_size) {
        printf("Error reading fc1 weights\n");
        fclose(f);
        return 0;
    }

    read = fread(cnn->fc1->bias, sizeof(float), cnn->fc1->output_size, f);
    if (read != cnn->fc1->output_size) {
        printf("Error reading fc1 biases\n");
        fclose(f);
        return 0;
    }

    // Load fc2 weights and biases
    int fc2_weights_size = cnn->fc2->input_size * cnn->fc2->output_size;
    read = fread(cnn->fc2->weights, sizeof(float), fc2_weights_size, f);
    if (read != fc2_weights_size) {
        printf("Error reading fc2 weights\n");
        fclose(f);
        return 0;
    }

    read = fread(cnn->fc2->bias, sizeof(float), cnn->fc2->output_size, f);
    if (read != cnn->fc2->output_size) {
        printf("Error reading fc2 biases\n");
        fclose(f);
        return 0;
    }

    fclose(f);
    printf("Model loaded from %s\n", filename);
    return 1;
}

// Test the model on a directory of images
void test_model(CNN* cnn, const char** test_image_paths, int num_test_images) {
    int correct = 0;
    int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};

    printf("Testing model on %d images...\n", num_test_images);

    for (int i = 0; i < num_test_images; i++) {
        // Load image
        load_image(test_image_paths[i], cnn->input);

        // Get ground truth label
        int true_label = parse_label(test_image_paths[i]);
        if (true_label < 0) continue;  // Skip if label can't be determined

        // Forward pass
        cnn_forward(cnn);

        // Find class with highest probability
        int predicted_class = 0;
        float max_prob = cnn->softmax_output[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (cnn->softmax_output[c] > max_prob) {
                max_prob = cnn->softmax_output[c];
                predicted_class = c;
            }
        }

        // Update confusion matrix
        confusion_matrix[true_label][predicted_class]++;

        // Update accuracy
        if (predicted_class == true_label) {
            correct++;
        }

        // Print progress
        if (i % 100 == 0) {
            printf("Tested %d/%d images\n", i + 1, num_test_images);
        }
    }

    // Calculate accuracy
    float accuracy = (float)correct / num_test_images;
    printf("Test accuracy: %.2f%%\n", accuracy * 100.0f);

    // Print confusion matrix
    printf("\nConfusion Matrix:\n");
    printf("%-10s", "");  // Empty cell for the top-left
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("%-10s", CLASS_NAMES[i]);
    }
    printf("\n");

    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("%-10s", CLASS_NAMES[i]);
        for (int j = 0; j < NUM_CLASSES; j++) {
            printf("%-10d", confusion_matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to read image paths from a directory
// Function to read image paths from a directory
int read_directory(const char* dir_path, char*** paths, const char** class_names, int num_classes) {
    int total_images = 0;
    *paths = NULL;

    // First count the total number of images
    for (int class_idx = 0; class_idx < num_classes; class_idx++) {
        char class_path[1024];
        snprintf(class_path, sizeof(class_path), "%s/%s", dir_path, class_names[class_idx]);

#ifdef _WIN32
        // Windows implementation
        WIN32_FIND_DATA findFileData;
        char searchPath[1024];
        snprintf(searchPath, sizeof(searchPath), "%s\\*", class_path);

        HANDLE hFind = FindFirstFile(searchPath, &findFileData);
        if (hFind == INVALID_HANDLE_VALUE) {
            printf("Could not open directory: %s\n", class_path);
            continue;
        }

        do {
            const char* ext = strrchr(findFileData.cFileName, '.');
            if (ext != NULL && (
                _stricmp(ext, ".jpg") == 0 ||
                _stricmp(ext, ".jpeg") == 0 ||
                _stricmp(ext, ".png") == 0)) {
                total_images++;
            }
        } while (FindNextFile(hFind, &findFileData) != 0);
        FindClose(hFind);
#else
        // Linux/macOS implementation
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(class_path)) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                const char* ext = strrchr(ent->d_name, '.');
                if (ext != NULL && (
                        strcasecmp(ext, ".jpg") == 0 ||
                        strcasecmp(ext, ".jpeg") == 0 ||
                        strcasecmp(ext, ".png") == 0)) {
                    total_images++;
                }
            }
            closedir(dir);
        } else {
            printf("Could not open directory: %s\n", class_path);
            continue;
        }
#endif
    }

    if (total_images == 0) {
        printf("No images found in directory: %s\n", dir_path);
        return 0;
    }

    // Allocate memory for paths
    *paths = (char**)malloc(total_images * sizeof(char*));
    if (*paths == NULL) {
        printf("Memory allocation failed\n");
        return 0;
    }

    // Now actually store the paths
    int current_idx = 0;
    for (int class_idx = 0; class_idx < num_classes; class_idx++) {
        char class_path[1024];
        snprintf(class_path, sizeof(class_path), "%s/%s", dir_path, class_names[class_idx]);

#ifdef _WIN32
        // Windows implementation
        WIN32_FIND_DATA findFileData;
        char searchPath[1024];
        snprintf(searchPath, sizeof(searchPath), "%s\\*", class_path);

        HANDLE hFind = FindFirstFile(searchPath, &findFileData);
        if (hFind == INVALID_HANDLE_VALUE) continue;

        do {
            const char* ext = strrchr(findFileData.cFileName, '.');
            if (ext != NULL && (
                _stricmp(ext, ".jpg") == 0 ||
                _stricmp(ext, ".jpeg") == 0 ||
                _stricmp(ext, ".png") == 0)) {

                (*paths)[current_idx] = (char*)malloc(1024);
                snprintf((*paths)[current_idx], 1024, "%s/%s", class_path, findFileData.cFileName);
                current_idx++;
            }
        } while (FindNextFile(hFind, &findFileData) != 0);
        FindClose(hFind);
#else
        // Linux/macOS implementation
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(class_path)) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                const char* ext = strrchr(ent->d_name, '.');
                if (ext != NULL && (
                        strcasecmp(ext, ".jpg") == 0 ||
                        strcasecmp(ext, ".jpeg") == 0 ||
                        strcasecmp(ext, ".png") == 0)) {

                    (*paths)[current_idx] = (char*)malloc(1024);
                    snprintf((*paths)[current_idx], 1024, "%s/%s", class_path, ent->d_name);
                    current_idx++;
                }
            }
            closedir(dir);
        }
#endif
    }

    return total_images;
}

// Main function
int main(int argc, char** argv) {
    // Seed random number generator
    srand(time(NULL));

    // Create the CNN
    CNN* cnn = create_cnn();

    // Define default paths for training, testing, and model saving
    const char* training_dir = "Animals/train";  // Default training directory
    const char* testing_dir = "Animals/test";    // Default testing directory
    const char* model_path = "animal_model.bin";   // Default model file path

    printf("Starting automatic training and testing process...\n");

    // TRAINING PHASE
    printf("\n--- TRAINING PHASE ---\n");

    // Read training data
    printf("Loading training data from %s...\n", training_dir);

    // Store image paths for training data
    char** image_paths = NULL;
    int num_images = read_directory(training_dir, &image_paths, CLASS_NAMES, NUM_CLASSES);

    printf("Found %d training images\n", num_images);

    if (num_images == 0) {
        printf("Error: No training images found!\n");
        printf("Please ensure training images exist in %s directory\n", training_dir);
        free_cnn(cnn);
        return 1;
    }

    // Train the model
    printf("Training model...\n");
    train_cnn(cnn, (const char**)image_paths, num_images);

    // Save the trained model
    printf("Saving model to %s...\n", model_path);
    save_model(cnn, model_path);

    // Clean up training data
    for (int i = 0; i < num_images; i++) {
        free(image_paths[i]);
    }
    free(image_paths);

    // TESTING PHASE
    printf("\n--- TESTING PHASE ---\n");

    // Load the trained model
    printf("Loading model from %s...\n", model_path);
    if (!load_model(cnn, model_path)) {
        printf("Error: Failed to load model!\n");
        free_cnn(cnn);
        return 1;
    }

    // Read test data
    printf("Loading test data from %s...\n", testing_dir);

    // Store image paths for test data
    char** test_image_paths = NULL;
    int num_test_images = read_directory(testing_dir, &test_image_paths, CLASS_NAMES, NUM_CLASSES);

    printf("Found %d test images\n", num_test_images);

    if (num_test_images == 0) {
        printf("Error: No test images found!\n");
        printf("Please ensure test images exist in %s directory\n", testing_dir);
        free_cnn(cnn);
        return 1;
    }

    // Test the model
    printf("Testing model...\n");
    test_model(cnn, (const char**)test_image_paths, num_test_images);

    // Clean up test data
    for (int i = 0; i < num_test_images; i++) {
        free(test_image_paths[i]);
    }
    free(test_image_paths);

    // Demonstrate prediction on a specific image
//    printf("\n--- SAMPLE PREDICTION ---\n");
//    const char* sample_image = "./sample_animal.jpg";
//    printf("Predicting sample image: %s\n", sample_image);
//    int predicted_class = predict(cnn, sample_image);
//    printf("Prediction: %s\n", CLASS_NAMES[predicted_class]);
//
//    // Clean up CNN
//    free_cnn(cnn);

    printf("\nProcess completed successfully!\n");
    return 0;
}

