#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define NUM_NODES 10
#define NUM_FEATURES 200
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.01
#define GRADIENT_CLIP 1.0f

typedef struct {
    int* neighbors;
    int num_neighbors;
} Node;

typedef struct {
    float** attention_weights;  // (NUM_FEATURES x NUM_FEATURES)
    float* attention_bias;      // (NUM_FEATURES)
    float** weights;            // (NUM_FEATURES x NUM_FEATURES)
} GNNParameters;

// -------------------- Activation and Norm Functions --------------------

// ReLU activation
void relu(float* features, int size) {
    for (int i = 0; i < size; i++) {
        if (features[i] < 0)
            features[i] = 0;
    }
}

// Sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Layer normalization
void layer_norm(float* features, int size) {
    float mean = 0.0f;
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += features[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++) {
        variance += (features[i] - mean) * (features[i] - mean);
    }
    variance /= size;
    float std_dev = sqrtf(variance + 1e-8f);
    for (int i = 0; i < size; i++) {
        features[i] = (features[i] - mean) / std_dev;
    }
}

// -------------------- File I/O Functions --------------------

// Read a matrix from a file (elements stored row-wise)
float** read_matrix(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening matrix file");
        exit(EXIT_FAILURE);
    }
    float** matrix = (float**)malloc(NUM_FEATURES * sizeof(float*));
    for (int i = 0; i < NUM_FEATURES; i++) {
        matrix[i] = (float*)malloc(NUM_FEATURES * sizeof(float));
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (fscanf(file, "%f", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading matrix element [%d][%d] from %s\n", i, j, filename);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
    return matrix;
}

// Read a vector from a file
float* read_vector(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening vector file");
        exit(EXIT_FAILURE);
    }
    float* vector = (float*)malloc(NUM_FEATURES * sizeof(float));
    for (int i = 0; i < NUM_FEATURES; i++) {
        if (fscanf(file, "%f", &vector[i]) != 1) {
            fprintf(stderr, "Error reading vector element [%d] from %s\n", i, filename);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
    return vector;
}

// Initialize parameters from files
void initialize_parameters_from_file(GNNParameters* params) {
    params->attention_weights = read_matrix("attention_weights.txt");
    params->weights = read_matrix("weights.txt");
    params->attention_bias = read_vector("attention_bias.txt");
}

// Allocate and zero-initialize gradients
void initialize_gradients(GNNParameters* gradients) {
    gradients->attention_weights = (float**)malloc(NUM_FEATURES * sizeof(float*));
    gradients->weights = (float**)malloc(NUM_FEATURES * sizeof(float*));
    gradients->attention_bias = (float*)malloc(NUM_FEATURES * sizeof(float));
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        gradients->attention_weights[i] = (float*)calloc(NUM_FEATURES, sizeof(float));
        gradients->weights[i] = (float*)calloc(NUM_FEATURES, sizeof(float));
    }
    for (int i = 0; i < NUM_FEATURES; i++) {
        gradients->attention_bias[i] = 0.0f;
    }
}

// -------------------- GNN Functions --------------------

// Compute attention score between two nodes with scaling
float attention_score(const float* node_i, const float* node_j, const GNNParameters* params) {
    float final_score = 0.0f;
    
    float local_score = 0.0f;
    for (int k = 0; k < NUM_FEATURES; k++) {
        float tempScore = node_i[k] * params->attention_bias[k];
        for (int l = 0; l < NUM_FEATURES; l++) {
            tempScore += node_i[k] * params->attention_weights[k][l] * node_j[l];
        }
        local_score += tempScore;
    }
    final_score += local_score;

    return final_score / sqrtf(NUM_FEATURES);
}

// Softmax function for attention scores
void softmax(float* scores, int size) {
    float max_score = scores[0];
    for (int i = 1; i < size; i++) {
        if (scores[i] > max_score)
            max_score = scores[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        scores[i] = expf(fminf(scores[i] - max_score, 10.0f));
        sum += scores[i];
    }
    if (sum < 1e-9)
        sum = 1e-9;
    for (int i = 0; i < size; i++) {
        scores[i] /= sum;
    }
}

// Forward pass: GNN layer with layer normalization
void gnn_layer(const Node* graph, float** node_features, const GNNParameters* params, float** cache) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_NODES; i++) {
        float* original_features = (float*)malloc(NUM_FEATURES * sizeof(float));
        memcpy(original_features, node_features[i], NUM_FEATURES * sizeof(float));

        if (graph[i].num_neighbors > 0) {
            float* scores = (float*)malloc(graph[i].num_neighbors * sizeof(float));
            float* aggregated = (float*)calloc(NUM_FEATURES, sizeof(float));

            // Calculate attention scores
            for (int j = 0; j < graph[i].num_neighbors; j++) {
                int neighbor = graph[i].neighbors[j];
                scores[j] = attention_score(original_features, node_features[neighbor], params);
            }

            // Apply softmax
            softmax(scores, graph[i].num_neighbors);

            // Cache attention scores for backward pass
            cache[i] = scores;

            // Aggregate neighbor features
            for (int j = 0; j < graph[i].num_neighbors; j++) {
                int neighbor = graph[i].neighbors[j];
                for (int k = 0; k < NUM_FEATURES; k++) {
                    aggregated[k] += scores[j] * node_features[neighbor][k];
                }
            }

            // Transform features and add to original
            for (int k = 0; k < NUM_FEATURES; k++) {
                for (int l = 0; l < NUM_FEATURES; l++) {
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

// Binary cross-entropy loss
float binary_cross_entropy(float pred, float target) {
    const float epsilon = 1e-9;
    pred = fmaxf(epsilon, fminf(pred, 1.0f - epsilon));
    return -target * logf(pred) - (1 - target) * logf(1 - pred);
}

// Backward pass: Compute gradients
void backward_pass(const Node* graph, float** node_features, const GNNParameters* params, GNNParameters* gradients, float** cache) {
    // Reset gradients to zero (gradients were allocated and zeroed already)
    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            gradients->weights[i][j] = 0.0f;
            gradients->attention_weights[i][j] = 0.0f;
        }
        gradients->attention_bias[i] = 0.0f;
    }

    // Compute gradients for each node
    for (int i = 0; i < NUM_NODES; i++) {
        if (graph[i].num_neighbors > 0) {
            float* scores = cache[i];

            // Gradients for attention weights and bias
            for (int j = 0; j < graph[i].num_neighbors; j++) {
                int neighbor = graph[i].neighbors[j];
                for (int k = 0; k < NUM_FEATURES; k++) {
                    gradients->attention_bias[k] += scores[j] * node_features[neighbor][k];
                    for (int l = 0; l < NUM_FEATURES; l++) {
                        gradients->attention_weights[k][l] += scores[j] * node_features[i][k] * node_features[neighbor][l];
                    }
                }
            }

            // Gradients for weight matrix
            for (int k = 0; k < NUM_FEATURES; k++) {
                for (int l = 0; l < NUM_FEATURES; l++) {
                    gradients->weights[k][l] += node_features[i][k] * node_features[i][l];
                }
            }
        }
    }
}

// Gradient clipping to prevent exploding gradients
void clip_gradients(GNNParameters* gradients) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            gradients->weights[i][j] = fmaxf(fminf(gradients->weights[i][j], GRADIENT_CLIP), -GRADIENT_CLIP);
            gradients->attention_weights[i][j] = fmaxf(fminf(gradients->attention_weights[i][j], GRADIENT_CLIP), -GRADIENT_CLIP);
        }
    }
    for (int i = 0; i < NUM_FEATURES; i++) {
        gradients->attention_bias[i] = fmaxf(fminf(gradients->attention_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }
}

// Update parameters using gradients
void update_parameters(GNNParameters* params, const GNNParameters* gradients) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            params->weights[i][j] -= LEARNING_RATE * gradients->weights[i][j];
            params->attention_weights[i][j] -= LEARNING_RATE * gradients->attention_weights[i][j];
        }
    }
    for (int i = 0; i < NUM_FEATURES; i++) {
        params->attention_bias[i] -= LEARNING_RATE * gradients->attention_bias[i];
    }
}

// Link prediction function
float predict_link(int node1, int node2, float** node_features, const GNNParameters* params) {
    float score = attention_score(node_features[node1], node_features[node2], params);
    return sigmoid(score);
}

// -------------------- Main Function --------------------

int main() {
    // Array of thread counts to test
    int thread_counts[] = {1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    // Open file to write results
    FILE *fp = fopen("benchmark_results.txt", "w");
    if (!fp) {
        perror("Error opening benchmark_results.txt");
        return 1;
    }
    
    
    for (int test = 0; test < num_tests; test++) {
        int num_threads = thread_counts[test];
        omp_set_num_threads(num_threads);
        
        srand(42);  // Use same seed for fair comparison
        
        // Initialize graph
        Node* graph = (Node*)malloc(NUM_NODES * sizeof(Node));
        for (int i = 0; i < NUM_NODES; i++) {
            graph[i].num_neighbors = (rand() % 10) + 1;
            graph[i].neighbors = (int*)malloc(graph[i].num_neighbors * sizeof(int));
            for (int j = 0; j < graph[i].num_neighbors; j++) {
                graph[i].neighbors[j] = rand() % NUM_NODES;
            }
        }

        // Initialize node features
        float** node_features = (float**)malloc(NUM_NODES * sizeof(float*));
        for (int i = 0; i < NUM_NODES; i++) {
            node_features[i] = (float*)malloc(NUM_FEATURES * sizeof(float));
            for (int j = 0; j < NUM_FEATURES; j++) {
                node_features[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }
        }

        // Initialize parameters and gradients
        GNNParameters params, gradients;
        initialize_parameters_from_file(&params);
        initialize_gradients(&gradients);

        // Allocate cache
        float** cache = (float**)malloc(NUM_NODES * sizeof(float*));
        for (int i = 0; i < NUM_NODES; i++) {
            cache[i] = NULL;
        }

        // Training loop with timing
        double start_time = omp_get_wtime();
        
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            gnn_layer(graph, node_features, &params, cache);

            float loss = 0.0f;
            for (int i = 0; i < NUM_NODES; i++) {
                float pred = sigmoid(node_features[i][0]);
                float target = (i == 0) ? 1.0f : 0.0f;
                loss += binary_cross_entropy(pred, target);
            }
            loss /= NUM_NODES;

            backward_pass(graph, node_features, &params, &gradients, cache);
            clip_gradients(&gradients);
            update_parameters(&params, &gradients);
        }
        
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        
        // Write results to file
        fprintf(fp, "%.6f\n", elapsed_time);
        printf("Completed benchmark with %d threads: %.6f seconds\n", num_threads, elapsed_time);

        // Cleanup for this iteration
        for (int i = 0; i < NUM_NODES; i++) {
            free(graph[i].neighbors);
            free(node_features[i]);
            if (cache[i])
                free(cache[i]);
        }
        free(graph);
        free(node_features);
        free(cache);

        for (int i = 0; i < NUM_FEATURES; i++) {
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
    }
    
    fclose(fp);
    return 0;
}