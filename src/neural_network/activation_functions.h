#pragma once

#include "linear_algebra/vector.h"

/// Activation functions σ and their derivatives σ′.
///
/// Activations are applied in place: the vector of inner potentials ξ is
/// overwritten with the outputs σ(ξ).
/// Derivatives read the potentials ξ and write σ′(ξ) into a preallocated
/// vector of the same size, so backpropagation can compute δ = g ⊙ σ′(ξ).
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
void Sigmoid(linalg::Vector &neurons);
/// σ′(ξ) = σ(ξ) · (1 − σ(ξ))
void SigmoidDerivative(const linalg::Vector &potentials, linalg::Vector &derivatives);

/// Hyperbolic tangent: σ(ξ) = tanh(ξ)
void Tanh(linalg::Vector &neurons);
/// σ′(ξ) = 1 − tanh²(ξ)
void TanhDerivative(const linalg::Vector &potentials, linalg::Vector &derivatives);

/// Rectified linear unit: σ(ξ) = max(0, ξ)
void ReLu(linalg::Vector &neurons);
/// σ′(ξ) = 1 if ξ > 0, otherwise 0
void ReLuDerivative(const linalg::Vector &potentials, linalg::Vector &derivatives);

/// Leaky ReLU: σ(ξ) = ξ if ξ > 0, otherwise α · ξ
void LeakyReLu(linalg::Vector &neurons, double alpha = 0.01);
/// σ′(ξ) = 1 if ξ > 0, otherwise α
void LeakyReLuDerivative(const linalg::Vector &potentials, linalg::Vector &derivatives, double alpha = 0.01);

/// Softmax for the output layer: σ(ξ)ᵢ = e^(ξᵢ) / Σⱼ e^(ξⱼ)
void Softmax(linalg::Vector &neurons);

/// Applies the activation function of the given type to neurons, in place
void Apply(Type type, linalg::Vector &neurons);

} // namespace actifunc
