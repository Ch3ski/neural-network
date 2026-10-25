#include "network.h"

#include <cassert>
#include <utility>

namespace nn
{

Network::Network(std::vector<Layer> layers)
    : m_layers(std::move(layers))
{
    assert(!m_layers.empty() && "Network: needs at least one layer");
}

const linalg::Vector &Network::output() const
{
    return m_layers.back().outputs();
}

void Network::forward(const linalg::Vector &input)
{
    const linalg::Vector *currentInput = &input;

    for (Layer &layer : m_layers) {
        layer.forward(*currentInput);
        currentInput = &layer.outputs();
    }
}

void Network::backward(const linalg::Vector &input, const linalg::Vector &target)
{
    assert(target.size() == output().size() && "Network: target size differs from the number of output neurons");

    for (auto layer = m_layers.rbegin(); layer != m_layers.rend(); ++layer) {
        break;
    }
}

void Network::learningStep(const double learningRate)
{
}

} // namespace nn
