#include "neural_network/network.h"
#include "test_utils.h"

#include <cmath>
#include <vector>

using actifunc::Type;

/// @brief The second layer gets the outputs y of the first layer, not its
///        potentials ξ
static void test_evaluate_passes_outputs() {
  // Layer 1: x = [1, 1] gives ξ = [−1, 6] and y = ReLU(ξ) = [0, 6]
  nn::Layer first(Type::ReLu, linalg::Matrix{{1.0, -2.0}, {3.0, 4.0}},
                  linalg::Vector{0.0, -1.0});
  // Layer 2: adds its two inputs
  nn::Layer second(Type::ReLu, linalg::Matrix{{1.0, 1.0}}, linalg::Vector{0.0});
  nn::Network network(std::vector<nn::Layer>{first, second});

  network.evaluate(linalg::Vector{1.0, 1.0});
  CHECK(network.outputs().size() == 1);
  CHECK_NEAR(network.outputs()[0], 6.0); // 0 + 6, it would be 5 with ξ
}

/// @brief Hand-set XOR network, sigmoid everywhere, 2 → 2 → 1:
///        hidden neuron 0 is OR, hidden neuron 1 is NAND, the output is AND
static void test_evaluate_xor() {
  nn::Layer hidden(Type::Sigmoid, linalg::Matrix{{20.0, 20.0}, {-20.0, -20.0}},
                   linalg::Vector{-10.0, 30.0});
  nn::Layer output(Type::Sigmoid, linalg::Matrix{{20.0, 20.0}},
                   linalg::Vector{-30.0});
  nn::Network network(std::vector<nn::Layer>{hidden, output});

  // σ(±10) is 1e-4 or less away from 0 or 1
  network.evaluate(linalg::Vector{0.0, 0.0});
  CHECK(std::abs(network.outputs()[0] - 0.0) < 1e-4);
  network.evaluate(linalg::Vector{0.0, 1.0});
  CHECK(std::abs(network.outputs()[0] - 1.0) < 1e-4);
  network.evaluate(linalg::Vector{1.0, 0.0});
  CHECK(std::abs(network.outputs()[0] - 1.0) < 1e-4);
  network.evaluate(linalg::Vector{1.0, 1.0});
  CHECK(std::abs(network.outputs()[0] - 0.0) < 1e-4);
}

int main() {
  RUN(test_evaluate_passes_outputs);
  RUN(test_evaluate_xor);
  return test::summary();
}
