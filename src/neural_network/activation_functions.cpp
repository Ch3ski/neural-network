#include "neural_network/activation_functions.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace actifunc
{

void Sigmoid(const linalg::Vector &potentials, linalg::Vector &outputs)
{
    outputs.apply(potentials, [](double x) {
        return 1 / (1.0 + std::exp(-x));
    });
}

void SigmoidDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives)
{
    derivatives.apply(outputs, [](double y) {
        return y * (1.0 - y);
    });
}

void Tanh(const linalg::Vector &potentials, linalg::Vector &outputs)
{
    // std::tanh is overloaded (float, double, ...), so it is wrapper
    outputs.apply(potentials, [](double x) {
        return std::tanh(x);
    });
}

void TanhDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives)
{
    derivatives.apply(outputs, [](double y) {
        return 1.0 - std::pow(y, 2);
    });
}

void ReLu(const linalg::Vector &potentials, linalg::Vector &outputs)
{
    outputs.apply(potentials, [](double x) {
        return std::max(0.0, x);
    });
}

void ReLuDerivative(const linalg::Vector &output, linalg::Vector &derivatives)
{
    derivatives.apply(output, [](double y) {
        return y > 0.0 ? 1.0 : 0.0;
    });
}

void LeakyReLu(const linalg::Vector &potentials, linalg::Vector &outputs, double alpha)
{
    outputs.apply(potentials, [alpha](double x) {
        return x < 0.0 ? alpha * x : x;
    });
}

void LeakyReLuDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives, double alpha)
{
    derivatives.apply(outputs, [alpha](double y) {
        return y > 0.0 ? 1 : alpha;
    });
}

void Softmax(const linalg::Vector &potentials, linalg::Vector &outputs)
{
    if (potentials.size() == 0) {
        return;
    }

    // Shifting by the maximum does not change the result, but keeps every
    // exponent <= 0, so exp() cannot overflow to infinity.
    double max = potentials[0];
    for (std::size_t i = 1; i < potentials.size(); ++i) {
        max = std::max(max, potentials[i]);
    }

    double sum = 0.0;
    outputs.apply(potentials, [max, &sum](double x) {
        const double e = std::exp(x - max);
        sum += e;
        return e;
    });

    outputs /= sum;
}

void Apply(Type type, const linalg::Vector &potentials, linalg::Vector &outputs)
{
    switch (type) {
    case Type::Sigmoid:
        Sigmoid(potentials, outputs);
        return;

    case Type::Tanh:
        Tanh(potentials, outputs);
        return;

    case Type::ReLu:
        ReLu(potentials, outputs);
        return;

    case Type::LeakyReLu:
        LeakyReLu(potentials, outputs);
        return;

    case Type::Softmax:
        Softmax(potentials, outputs);
        return;
    }

    assert(false && "Apply: unknown activation type");
}

double Derivative(Type type, const double output)
{
    switch (type) {
    case Type::Sigmoid:
        return output * (1.0 - output);

    case Type::Tanh:
        return 1.0 - output * output;

    case Type::ReLu:
        return output > 0.0 ? 1.0 : 0.0;

    case Type::LeakyReLu:
        // Same α as the default of LeakyReLu
        return output > 0.0 ? 1.0 : 0.01;

    case Type::Softmax:
        assert(false && "Derivative: Softmax has no derivative per neuron");
        return 0.0;
    }

    assert(false && "Derivative: unknown activation type");
    return 0.0;
}

} // namespace actifunc
