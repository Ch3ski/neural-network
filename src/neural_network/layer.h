#pragma once

#include "activation_functions.h"
#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"

namespace nn
{

/// One fully connected layer: every neuron is connected to every input.
/// Neuron j's weights are row j of m_weights and its bias is m_bias[j].
class Layer
{
public:
    /// @brief Creates a layer from given weights, e.g. hand-set ones for tests
    /// @param activationType Activation function σ of every neuron
    /// @param weights        Weight matrix W, size neurons × inputs
    /// @param bias           Bias vector b, size neurons
    /// @pre bias.size() == weights.rows()
    explicit Layer(actifunc::Type activationType, linalg::Matrix weights, linalg::Vector bias);

    /// @brief Returns the outputs y = σ(ξ) of the last forward pass, read-only
    /// @return Const reference to the outputs, no copy
    const linalg::Vector &outputs() const;

    /// @brief Returns the inner potentials ξ = W · x + b of the last forward pass, read-only
    /// @return Const reference to the potentials, no copy
    const linalg::Vector &potentials() const;

    /// @brief Computes ξ = W · x + b and y = σ(ξ) for one input
    ///        Read the results with potentials() and outputs()
    /// @param input Input vector x, the previous layer's outputs or the image for the first layer
    /// @pre input.size() == number of columns of W
    void forward(const linalg::Vector &input);

private:
    /// Activation function type for σ and derivative σ′
    actifunc::Type m_activationType;

    /// Weight matrix W, size (neurons in this layer) × (inputs to this layer).
    linalg::Matrix m_weights;
    /// Bias vector b, one value per neuron in this layer.
    linalg::Vector m_bias;

    /// Inner potentials ξ = W · x + b from the last forward pass, one per neuron.
    linalg::Vector m_potentials;
    /// Outputs y = σ(ξ) from the last forward pass, one per neuron.
    linalg::Vector m_outputs;
};

inline const linalg::Vector &Layer::potentials() const
{
    return m_potentials;
}

inline const linalg::Vector &Layer::outputs() const
{
    return m_outputs;
}

} // namespace nn
