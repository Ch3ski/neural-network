#pragma once

#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"

namespace nn
{

/// One fully connected layer: every neuron is connected to every input.
/// Neuron j's weights are row j of m_weights and its bias is m_bias[j].
class Layer
{
public:

private:
    /// Weight matrix W, size (neurons in this layer) × (inputs to this layer).
    linalg::Matrix m_weights;
    /// Bias vector b, one value per neuron in this layer.
    linalg::Vector m_bias;
};

} // namespace nn
