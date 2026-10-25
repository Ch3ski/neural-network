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

    /// @brief Returns the outputs y of the last layer from the last forward(), read-only
    /// @return Const reference to the outputs, no copy
    const linalg::Vector &output() const;

    /// @brief Evaluates all layers in order: the first gets input, every next
    ///        one gets the outputs of the previous one
    ///        Read the result with output()
    /// @param input Input vector x
    void forward(const linalg::Vector &input);

    /// @brief Backpropagation for one sample.
    /// @param input  Input vector x of the last forward()
    /// @param target Target vector t
    /// @pre forward(input) was called right before, with the same input
    /// @pre target.size() == number of neurons in the last layer
    void backward(const linalg::Vector &input, const linalg::Vector &target);

    /// @brief Gradient descent step in every layer: W −= η · dW, b −= η · db,
    ///        then clears dW and db for the next backward()
    /// @param learningRate Learning rate η, the size of the step
    void learningStep(const double learningRate);

private:
    std::vector<Layer> m_layers;
};

} // namespace nn
