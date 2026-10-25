#include "neural_network/activation_functions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace actifunc
{

void Sigmoid(linalg::Vector &neurons)
{
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i] = 1.0 / (1.0 + std::exp(-neurons[i]));
    }
}

void Tanh(linalg::Vector &neurons)
{
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i] = std::tanh(neurons[i]);
    }
}

void ReLu(linalg::Vector &neurons)
{
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i] = std::max(0.0, neurons[i]);
    }
}

void LeakyReLu(linalg::Vector &neurons, double alpha)
{
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        if (neurons[i] < 0.0) {
            neurons[i] *= alpha;
        }
    }
}

void Softmax(linalg::Vector &neurons)
{
    if (neurons.size() == 0) {
        return;
    }

    // Shifting by the maximum does not change the result, but keeps every
    // exponent <= 0, so exp() cannot overflow to infinity.
    double max = neurons[0];
    for (std::size_t i = 1; i < neurons.size(); ++i) {
        max = std::max(max, neurons[i]);
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i] = std::exp(neurons[i] - max);
        sum += neurons[i];
    }

    for (std::size_t i = 0; i < neurons.size(); ++i) {
        neurons[i] /= sum;
    }
}

} // namespace actifunc
