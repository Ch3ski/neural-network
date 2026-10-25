#pragma once

#include "layer.h"

#include <vector>

namespace nn
{

/// Feed-forward neural network: an ordered list of layers, where the output
/// of each layer is the input of the next one.
class Network
{
public:
    /// @brief Creates a network from ready-made layers, e.g. hand-set ones for tests
    /// @param layers Layers in order from input to output, taken by value and moved in
    /// @pre layers is not empty
    explicit Network(std::vector<Layer> layers);

    /// @brief Evaluates all layers in order: the first gets input, every next
    ///        one gets the outputs of the previous one
    ///        Read the result with outputs()
    /// @param input Input vector x
    void forward(const linalg::Vector &input);

    /// @brief Returns the outputs of the last layer from the last evaluate, read-only
    /// @return Const reference to the outputs, no copy
    const linalg::Vector &outputs() const;

private:
    std::vector<Layer> m_layers;
};

inline const linalg::Vector &Network::outputs() const
{
    return m_layers.back().outputs();
}

} // namespace nn
