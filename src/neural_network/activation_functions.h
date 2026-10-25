#pragma once

#include "linear_algebra/vector.h"

/// Activation functions σ and their derivatives σ′.
///
/// Activations read the inner potentials ξ and write the outputs σ(ξ) into a
/// preallocated vector of the same size, so ξ stays unchanged for
/// backpropagation.
/// Derivatives read the outputs y = σ(ξ), not ξ, and write σ′(ξ) into a
/// preallocated vector of the same size, so backpropagation can compute
/// δ = g ⊙ σ′(ξ). For these activations σ′ can be written with y alone, so
/// no exp is needed.
namespace actifunc
{

/// Which activation function a layer uses.
enum class Type {
    Sigmoid,
    Tanh,
    ReLu,
    LeakyReLu,
    Softmax,
};

/// Logistic sigmoid: σ(ξ) = 1 / (1 + e^(−ξ))
void Sigmoid(const linalg::Vector &potentials, linalg::Vector &outputs);
/// σ′(ξ) = σ(ξ) · (1 − σ(ξ)) = y · (1 − y)
void SigmoidDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives);

/// Hyperbolic tangent: σ(ξ) = tanh(ξ)
void Tanh(const linalg::Vector &potentials, linalg::Vector &outputs);
/// σ′(ξ) = 1 − tanh²(ξ) = 1 − y²
void TanhDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives);

/// Rectified linear unit: σ(ξ) = max(0, ξ)
void ReLu(const linalg::Vector &potentials, linalg::Vector &outputs);
/// σ′(ξ) = 1 if y > 0, otherwise 0 (y > 0 exactly when ξ > 0)
void ReLuDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives);

/// Leaky ReLU: σ(ξ) = ξ if ξ > 0, otherwise α · ξ
void LeakyReLu(const linalg::Vector &potentials, linalg::Vector &outputs, double alpha = 0.01);
/// σ′(ξ) = 1 if y > 0, otherwise α (α > 0 keeps the sign of ξ in y)
void LeakyReLuDerivative(const linalg::Vector &outputs, linalg::Vector &derivatives, double alpha = 0.01);

/// Softmax for the output layer: σ(ξ)ᵢ = e^(ξᵢ) / Σⱼ e^(ξⱼ)
void Softmax(const linalg::Vector &potentials, linalg::Vector &outputs);

/// Applies the activation function of the given type: outputs = σ(potentials)
void Apply(Type type, const linalg::Vector &potentials, linalg::Vector &outputs);

/// σ′(ξ) of one neuron from its output y = σ(ξ)
/// Softmax has no σ′ per neuron and asserts
double Derivative(Type type, const double output);

} // namespace actifunc
