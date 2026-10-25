#include "neural_network/activation_functions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace actifunc
{

void Sigmoid(linalg::Vector &neurons)
{
    neurons.apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

void Tanh(linalg::Vector &neurons)
{
    // std::tanh is overloaded (float, double, ...), so it is wrapped in a
    // lambda instead of being passed directly.
    neurons.apply([](double x) { return std::tanh(x); });
}

void ReLu(linalg::Vector &neurons)
{
    neurons.apply([](double x) { return std::max(0.0, x); });
}

void LeakyReLu(linalg::Vector &neurons, double alpha)
{
    neurons.apply([alpha](double x) { return x < 0.0 ? alpha * x : x; });
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
    neurons.apply([max, &sum](double x) {
        const double e = std::exp(x - max);
        sum += e;
        return e;
    });

    neurons /= sum;
}

} // namespace actifunc
