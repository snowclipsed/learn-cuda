#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <algorithm>

using namespace std;

__global__ void count_pairs_kernel(char *vocab, int *vocab_counts, int vocab_size, int *pair_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        for (int i = 0; i < vocab_size; ++i) {
            if (i != idx && vocab[idx] == vocab[i]) {
                atomicAdd(&pair_counts[idx], vocab_counts[i]);
            }
        }
    }
}

vector<string> split(const string &str) {
    vector<string> tokens;
    istringstream iss(str);
    string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

string join(const vector<string> &tokens, const string &delimiter = " ") {
    string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += delimiter;
        result += tokens[i];
    }
    return result;
}

unordered_map<string, int> get_vocab(const vector<string> &corpus) {
    unordered_map<string, int> vocab;
    for (const string &sentence : corpus) {
        vector<string> words = split(sentence);
        for (string &word : words) {
            string spaced_word;
            for (char c : word) {
                spaced_word += c;
                spaced_word += ' ';
            }
            spaced_word += "</w>";
            vocab[spaced_word]++;
        }
    }
    return vocab;
}

unordered_map<string, int> merge_vocab(const pair<string, string> &pair, const unordered_map<string, int> &vocab) {
    unordered_map<string, int> new_vocab;
    string bigram = pair.first + " " + pair.second;
    string replacement = pair.first + pair.second;
    for (const auto &item : vocab) {
        string new_word = item.first;
        size_t pos;
        while ((pos = new_word.find(bigram)) != string::npos) {
            new_word.replace(pos, bigram.length(), replacement);
        }
        new_vocab[new_word] = item.second;
    }
    return new_vocab;
}

unordered_map<string, int> train_bpe(const vector<string> &corpus, int num_merges) {
    unordered_map<string, int> vocab = get_vocab(corpus);
    for (int i = 0; i < num_merges; ++i) {
        // Parallelized pair counting using CUDA
        int vocab_size = vocab.size();
        char *d_vocab;
        int *d_vocab_counts;
        int *d_pair_counts;

        // Allocate device memory
        cudaMalloc(&d_vocab, vocab_size * sizeof(char));
        cudaMalloc(&d_vocab_counts, vocab_size * sizeof(int));
        cudaMalloc(&d_pair_counts, vocab_size * sizeof(int));

        // Copy data to device
        cudaMemcpy(d_vocab, vocab.data(), vocab_size * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vocab_counts, vocab_counts.data(), vocab_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pair_counts, pair_counts.data(), vocab_size * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int blockSize = 256;
        int numBlocks = (vocab_size + blockSize - 1) / blockSize;
        count_pairs_kernel<<<numBlocks, blockSize>>>(d_vocab, d_vocab_counts, vocab_size, d_pair_counts);

        // Copy results back to host
        cudaMemcpy(pair_counts.data(), d_pair_counts, vocab_size * sizeof(int), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_vocab);
        cudaFree(d_vocab_counts);
        cudaFree(d_pair_counts);

        // Find the most frequent pair
        auto best_pair = max_element(pair_counts.begin(), pair_counts.end());

        // Merge the best pair
        vocab = merge_vocab(best_pair, vocab);
    }
    return vocab;
}

int main() {
    vector<string> corpus = {
        "this is a test sentence",
        "this is another sentence",
        "yet another sentence for testing"
    };

    int num_merges = 10;
    unordered_map<string, int> vocab = train_bpe(corpus, num_merges);

    string sentence = "this is a test";
    vector<string> words = split(sentence);
    for (const string &word : words) {
        vector<string> tokens = tokenize(word, vocab);
        for (const string &token : tokens) {
            cout << token << " ";
        }
    }
    cout << endl;

    return 0;
}
