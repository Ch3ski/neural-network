#include "neural_network/layer.h"

#include "linear_algebra/matrix_vector.h"

#include <cassert>
#include <utility>

namespace nn
{

Layer::Layer(actifunc::Type activationType, linalg::Matrix weights, linalg::Vector bias)
    : m_activationType{activationType}
    , m_weights(std::move(weights))
    , m_bias(std::move(bias))
    , m_potentials(m_weights.rows(), 0.0)
    , m_outputs(m_weights.rows(), 0.0)
{
    assert(m_bias.size() == m_weights.rows() && "Layer: bias size differs from the number of neurons");
}

void Layer::forward(const linalg::Vector &input)
{
    assert(input.size() == m_weights.cols() && "Layer: input size differs from the number of weight columns");

    linalg::Multiply(m_weights, input, m_potentials);
    m_potentials += m_bias;
    actifunc::Apply(m_activationType, m_potentials, m_outputs);
}

} // namespace nn
