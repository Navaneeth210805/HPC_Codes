#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

const int NUM_NODES = 1000;
const int NUM_FEATURES = 128;
const int NUM_EPOCHS = 30;

struct Node {
    vector<int> neighbors;
};

void relu(vector<float>& features) {
    for (float& f : features) {
        if (f < 0) f = 0;
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void initialize_node_features(vector<vector<float>>& features, int num_nodes, int num_features) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_features; j++) {
            features[i][j] = dis(gen);
        }
    }
}

void initialize_weight_matrix(vector<vector<float>>& weights, int num_features) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);

    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_features; j++) {
            weights[i][j] = dis(gen);
        }
    }
}

void matrix_multiply(const vector<float>& input, const vector<vector<float>>& weights, vector<float>& output) {
    int size = weights.size();
    for (int i = 0; i < size; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < size; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
}

vector<vector<float>> attention_weights(NUM_FEATURES, vector<float>(NUM_FEATURES, 0.0f));
vector<float> attention_bias(NUM_FEATURES, 0.0f);

void initialize_attention_parameters() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);

    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            attention_weights[i][j] = dis(gen);
        }
        attention_bias[i] = dis(gen);
    }
}

float attention_score(const vector<float>& node_i, const vector<float>& node_j) {
    float score = 0.0f;
    for (int k = 0; k < NUM_FEATURES; ++k) {
        for (int l = 0; l < NUM_FEATURES; ++l) {
            score += node_i[k] * attention_weights[k][l] * node_j[l];
        }
        score += attention_bias[k];
    }
    return sigmoid(score);
}

void gnn_layer(const vector<Node>& graph, vector<vector<float>>& node_features, int num_nodes, const vector<vector<float>>& weights) {
    vector<vector<float>> updated_features(num_nodes, vector<float>(NUM_FEATURES, 0.0f));

    for (int i = 0; i < num_nodes; i++) {
        vector<float> aggregated_features(NUM_FEATURES, 0.0f);

        for (int neighbor : graph[i].neighbors) {
            float attn = attention_score(node_features[i], node_features[neighbor]);
            for (int k = 0; k < NUM_FEATURES; k++) {
                aggregated_features[k] += attn * node_features[neighbor][k];
            }
        }

        matrix_multiply(aggregated_features, weights, updated_features[i]);
        relu(updated_features[i]);

        for (int k = 0; k < NUM_FEATURES; ++k) {
            updated_features[i][k] += node_features[i][k];
        }
    }

    node_features = updated_features;
}

float link_prediction(const vector<float>& node_u, const vector<float>& node_v) {
    float dot_product = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        dot_product += node_u[i] * node_v[i];
    }
    return sigmoid(dot_product);
}

int main() {
    vector<Node> graph(NUM_NODES);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, NUM_NODES - 1);

    for (int i = 0; i < NUM_NODES; i++) {
        int num_neighbors = dis(gen) % 10 + 1; 
        for (int j = 0; j < num_neighbors; j++) {
            int neighbor = dis(gen);
            if (neighbor != i) {
                graph[i].neighbors.push_back(neighbor);
            }
        }
    }

    vector<vector<float>> node_features(NUM_NODES, vector<float>(NUM_FEATURES, 0.0f));
    initialize_node_features(node_features, NUM_NODES, NUM_FEATURES);

    vector<vector<float>> weights(NUM_FEATURES, vector<float>(NUM_FEATURES, 0.0f));
    initialize_weight_matrix(weights, NUM_FEATURES);

    initialize_attention_parameters();

    cout << "Running GNN with " << NUM_NODES << " nodes and " << NUM_FEATURES << " features...\n";

    auto start = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        gnn_layer(graph, node_features, NUM_NODES, weights);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    int u = 0, v = NUM_NODES - 1;
    float score = link_prediction(node_features[u], node_features[v]);
    cout << "Link prediction score between nodes " << u << " and " << v << ": " << score << endl;

    return 0;
}