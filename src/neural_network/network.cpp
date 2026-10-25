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

void Network::evaluate(const linalg::Vector &input)
{
    const linalg::Vector *currentInput = &input;

    for (Layer &layer : m_layers) {
        layer.evaluate(*currentInput);
        currentInput = &layer.outputs();
    }
}

} // namespace nn
