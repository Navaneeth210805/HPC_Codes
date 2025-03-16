#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define NUM_NODES 10
#define NUM_FEATURES 200
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.01
#define GRADIENT_CLIP 1.0f

typedef struct
{
    int *neighbors;
    int num_neighbors;
} Node;

typedef struct
{
    float *attention_weights;
    float *attention_bias;
    float *weights;
} GNNParameters;

void relu(float *features, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (features[i] < 0)
            features[i] = 0;
    }
}

void layer_norm(float *features, int size)
{
    float mean = 0.0f;
    float variance = 0.0f;
    for (int i = 0; i < size; i++)
    {
        mean += features[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++)
    {
        variance += (features[i] - mean) * (features[i] - mean);
    }
    variance /= size;
    float std_dev = sqrtf(variance + 1e-8f);
    for (int i = 0; i < size; i++)
    {
        features[i] = (features[i] - mean) / std_dev;
    }
}

float *read_matrix(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening matrix file");
        exit(EXIT_FAILURE);
    }
    float *matrix = (float *)malloc(NUM_FEATURES * NUM_FEATURES * sizeof(float));
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        int index = i * NUM_FEATURES;
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            if (fscanf(file, "%f", &matrix[index + j]) != 1)
            {
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
    return matrix;
}

float *read_vector(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening vector file");
        exit(EXIT_FAILURE);
    }
    float *vector = (float *)malloc(NUM_FEATURES * sizeof(float));
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        if (fscanf(file, "%f", &vector[i]) != 1)
        {
            fprintf(stderr, "Error reading vector element [%d] from %s\n", i, filename);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
    return vector;
}

void initialize_parameters_from_file(GNNParameters *params)
{
    params->attention_weights = read_matrix("attention_weights.txt");
    params->weights = read_matrix("weights.txt");
    params->attention_bias = read_vector("attention_bias.txt");
}

void initialize_gradients(GNNParameters *gradients)
{
    gradients->attention_weights = (float *)calloc(NUM_FEATURES * NUM_FEATURES ,sizeof(float));
    gradients->weights = (float *)calloc(NUM_FEATURES * NUM_FEATURES ,sizeof(float));
    gradients->attention_bias = (float *)calloc(NUM_FEATURES ,sizeof(float));

}

float attention_score(const float *node_i, const float *node_j, const GNNParameters *params)
{
    float final_score = 0.0f;

    float local_score = 0.0f;
    for (int k = 0; k < NUM_FEATURES; k++)
    {
        float tempScore = node_i[k] * params->attention_bias[k];
        for (int l = 0; l < NUM_FEATURES; l++)
        {
            tempScore += node_i[k] * params->attention_weights[k][l] * node_j[l];
        }
        local_score += tempScore;
    }
    final_score += local_score;

    return final_score / sqrtf(NUM_FEATURES);
}

void softmax(float *scores, int size)
{
    float max_score = scores[0];
    for (int i = 1; i < size; i++)
    {
        if (scores[i] > max_score)
            max_score = scores[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        scores[i] = expf(fminf(scores[i] - max_score, 10.0f));
        sum += scores[i];
    }
    if (sum < 1e-9)
        sum = 1e-9;
    for (int i = 0; i < size; i++)
    {
        scores[i] /= sum;
    }
}

void gnn_layer(const Node *graph, float **node_features, const GNNParameters *params, float **cache)
{
    for (int i = 0; i < NUM_NODES; i++)
    {
        float *original_features = (float *)malloc(NUM_FEATURES * sizeof(float));
        memcpy(original_features, node_features[i], NUM_FEATURES * sizeof(float));

        if (graph[i].num_neighbors > 0)
        {
            float *scores = (float *)malloc(graph[i].num_neighbors * sizeof(float));
            float *aggregated = (float *)calloc(NUM_FEATURES, sizeof(float));

            // Calculate attention scores
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                scores[j] = attention_score(original_features, node_features[neighbor], params);
            }

            // Apply softmax
            softmax(scores, graph[i].num_neighbors);

            // Cache attention scores for backward pass
            cache[i] = scores;

            // Aggregate neighbor features
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                for (int k = 0; k < NUM_FEATURES; k++)
                {
                    aggregated[k] += scores[j] * node_features[neighbor][k];
                }
            }

            // Transform features and add to original
            for (int k = 0; k < NUM_FEATURES; k++)
            {
                for (int l = 0; l < NUM_FEATURES; l++)
                {
                    node_features[i][k] += aggregated[k] * params->weights[k][l];
                }
            }

            free(aggregated);
        }

        // Apply layer normalization and ReLU
        layer_norm(node_features[i], NUM_FEATURES);
        relu(node_features[i], NUM_FEATURES);

        free(original_features);
    }
}

void backward_pass(const Node *graph, float **node_features, const GNNParameters *params, GNNParameters *gradients, float **cache)
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            gradients->weights[i][j] = 0.0f;
            gradients->attention_weights[i][j] = 0.0f;
        }
        gradients->attention_bias[i] = 0.0f;
    }

    for (int i = 0; i < NUM_NODES; i++)
    {
        if (graph[i].num_neighbors > 0)
        {
            float *scores = cache[i];

            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                for (int k = 0; k < NUM_FEATURES; k++)
                {
                    gradients->attention_bias[k] += scores[j] * node_features[neighbor][k];
                    for (int l = 0; l < NUM_FEATURES; l++)
                    {
                        gradients->attention_weights[k][l] += scores[j] * node_features[i][k] * node_features[neighbor][l];
                    }
                }
            }

            for (int k = 0; k < NUM_FEATURES; k++)
            {
                for (int l = 0; l < NUM_FEATURES; l++)
                {
                    gradients->weights[k][l] += node_features[i][k] * node_features[i][l];
                }
            }
        }
    }
}

void clip_gradients(GNNParameters *gradients)
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            gradients->weights[i*NUM_FEATURES + j] = fmaxf(fminf(gradients->weights[i*NUM_FEATURES + j], GRADIENT_CLIP), -GRADIENT_CLIP);
            gradients->attention_weights[i*NUM_FEATURES + j] = fmaxf(fminf(gradients->attention_weights[i*NUM_FEATURES + j], GRADIENT_CLIP), -GRADIENT_CLIP);
        }
    }
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        gradients->attention_bias[i] = fmaxf(fminf(gradients->attention_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float predict_link(int node1, int node2, float **node_features, const GNNParameters *params)
{
    float score = attention_score(node_features[node1], node_features[node2], params);
    return sigmoid(score);
}


__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float binary_cross_entropy(float pred, float target)
{
    return -(target * logf(pred + 1e-8) + (1.0f - target) * logf(1.0f - pred + 1e-8));
}

__global__ void compute_loss(float *d_node_features, float *d_loss)
{
    __shared__ float local_sum;
    int node_idx = blockIdx.x;
    int feature_idx = threadIdx.x
    if (node_idx >= NUM_NODES || feature_idx >= NUM_FEATURES) return;
    if(threadIdx.x == 0){
        local_sum = 0.0f;
    }
    __syncthreads();
    atomicAdd(&local_sum, d_node_features[node_idx*NUM_FEATURES + feature_idx]);
    __syncthreads();
    if(threadIdx.x == 0){
        float pred = sigmoid(local_sum);
        float target = (i == 0) ? 1.0f : 0.0f;
        float local_loss = binary_cross_entropy(pred, target);
        atomicAdd(loss, local_loss);
    }

}

__global__ void update_parameters(float *d_params_attention_bias, float *d_params_attention_weights, float *d_params_weights, float *d_grad_attention_bias, float *d_grad_attention_weights, float *d_grad_weights)
{
    int node_idx = blockIdx.x;
    int feature_idx = threadIdx.x;
    if (node_idx >= NUM_NODES || feature_idx >= NUM_FEATURES) return;
    d_params_attention_bias[feature_idx] -= LEARNING_RATE * d_grad_attention_bias[feature_idx];
    d_params_attention_weights[node_idx * NUM_FEATURES + feature_idx] -= LEARNING_RATE * d_grad_attention_weights[node_idx * NUM_FEATURES + feature_idx];
    d_params_weights[node_idx * NUM_FEATURES + feature_idx] -= LEARNING_RATE * d_grad_weights[node_idx * NUM_FEATURES + feature_idx];
}

int main()
{
    srand(42);

    Node *graph = (Node *)malloc(NUM_NODES * sizeof(Node));
    for (int i = 0; i < NUM_NODES; i++)
    {
        graph[i].num_neighbors = (rand() % 10) + 1;
        graph[i].neighbors = (int *)malloc(graph[i].num_neighbors * sizeof(int));
        for (int j = 0; j < graph[i].num_neighbors; j++)
        {
            graph[i].neighbors[j] = rand() % NUM_NODES;
        }
    }

    float *node_features = (float *)malloc(NUM_NODES * NUM_FEATURES * sizeof(float));
    for (int i = 0; i < NUM_NODES; i++)
    {
        int index = i * NUM_NODES;
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            node_features[index+j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    float *d_node_features;
    float *d_grad_weights;
    float *d_grad_attention_weights;
    float *d_grad_attention_bias;
    float *d_params_attention_weights;
    float *d_params_attention_bias;
    float *d_params_weights;
    cudaMalloc(&d_node_features, NUM_NODES * NUM_FEATURES * sizeof(float));
    cudaMalloc(&d_grad_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float));
    cudaMalloc(&d_grad_attention_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float));
    cudaMalloc(&d_grad_attention_bias, NUM_FEATURES * sizeof(float));
    cudaMalloc(&d_params_attention_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float));
    cudaMalloc(&d_params_attention_bias, NUM_FEATURES * sizeof(float));
    cudaMalloc(&d_params_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float));

    GNNParameters params, gradients;
    initialize_parameters_from_file(&params);
    initialize_gradients(&gradients);

    float **cache = (float **)malloc(NUM_NODES * sizeof(float *));
    for (int i = 0; i < NUM_NODES; i++)
    {
        cache[i] = NULL;
    }

    double start_time = clock();

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        gnn_layer(graph, node_features, &params, cache);
        
        cudaMemcpy(d_node_features, node_features, NUM_NODES * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        float *d_loss;
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));
        float loss = 0.0f;
        compute_loss<<<NUM_NODES,NUM_FEATURES>>>(d_node_features, d_loss);
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        loss /= NUM_NODES;
        backward_pass(graph, node_features, &params, &gradients, cache);
        clip_gradients(&gradients);
        cudaMemcpy(d_grad_weights, gradients.weights, NUM_FEATURES * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_attention_weights, gradients.attention_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_attention_bias, gradients.attention_bias, NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params_attention_weights, params.attention_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params_attention_bias, params.attention_bias, NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_params_weights, params.weights, NUM_FEATURES * NUM_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
        update_parameters<<<NUM_NODES,NUM_FEATURES>>>(d_params_attention_bias, d_params_attention_weights, d_params_weights, d_grad_attention_bias, d_grad_attention_weights, d_grad_weights);
        cudaMemcpy(params.attention_weights, d_params_attention_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(params.attention_bias, d_params_attention_bias, NUM_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(params.weights, d_params_weights, NUM_FEATURES * NUM_FEATURES * sizeof(float), cudaMemcpyDeviceToHost);
    }

    double end_time = clock();
    double elapsed_time = end_time - start_time;
    printf("Time taken: %f seconds\n", elapsed_time);

    for (int i = 0; i < NUM_NODES; i++)
    {
        free(graph[i].neighbors);
        free(node_features[i]);
        if (cache[i])
            free(cache[i]);
    }
    free(graph);
    free(node_features);
    free(cache);

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        free(params.attention_weights[i]);
        free(params.weights[i]);
        free(gradients.attention_weights[i]);
        free(gradients.weights[i]);
    }
    free(params.attention_weights);
    free(params.attention_bias);
    free(params.weights);
    free(gradients.attention_weights);
    free(gradients.attention_bias);
    free(gradients.weights);

    fclose(fp);
    return 0;
}