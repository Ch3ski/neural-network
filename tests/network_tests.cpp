#include "neural_network/network.h"
#include "test_utils.h"

#include <cmath>
#include <cstddef>
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

/// @brief Real topology 784 → 128 → 10: hidden W = 1/784 everywhere, so with
///        x = all 1 every hidden output is 1; output W(row, ·) = row / 128, so
///        ξ = [0, 1, …, 9] and y[row] = e^row / Σ e^j
static void test_evaluate_large_network() {
  nn::Layer hidden(Type::ReLu, linalg::Matrix(128, 784, 1.0 / 784.0),
                   linalg::Vector(128, 0.0));
  linalg::Matrix outputWeights(10, 128, 0.0);
  for (std::size_t row = 0; row < outputWeights.rows(); ++row) {
    for (std::size_t column = 0; column < outputWeights.cols(); ++column) {
      outputWeights(row, column) = static_cast<double>(row) / 128.0;
    }
  }
  nn::Layer output(Type::Softmax, outputWeights, linalg::Vector(10, 0.0));
  nn::Network network(std::vector<nn::Layer>{hidden, output});
  const linalg::Vector x(784, 1.0);

  test::benchmark("Network evaluate 784 -> 128 -> 10", 200,
                  [&] { network.evaluate(x); });

  double sum = 0.0;
  for (std::size_t j = 0; j < 10; ++j) {
    sum += std::exp(static_cast<double>(j));
  }
  for (std::size_t row = 0; row < 10; ++row) {
    CHECK_NEAR(network.outputs()[row],
               std::exp(static_cast<double>(row)) / sum);
  }
}

/// @brief 2 → 2 (ReLU) → 2 (Softmax) network for the learning tests, the
///        hidden layer is the one from test_evaluate_passes_outputs
static nn::Network learning_network() {
  nn::Layer hidden(Type::ReLu, linalg::Matrix{{1.0, -2.0}, {3.0, 4.0}},
                   linalg::Vector{0.0, -1.0});
  nn::Layer output(Type::Softmax, linalg::Matrix{{2.0, 0.5}, {1.0, -1.0}},
                   linalg::Vector{-3.0, 6.0});
  return nn::Network(std::vector<nn::Layer>{hidden, output});
}

/// @brief Checks the outputs for x = [1, 1] after one learning step with
///        x = [1, 1], t = [1, 0], η = 0.1:
///        forward  ξ1 = [−1, 6], y1 = [0, 6], ξ2 = [0, 0], y2 = [0.5, 0.5]
///        backward δ2 = y2 − t = [−0.5, 0.5], W2ᵀ · δ2 = [−0.5, −0.75],
///                 δ1 = [−0.5, −0.75] ⊙ ReLU′(ξ1) = [0, −0.75]
///        update   W1 = [[1, −2], [3.075, 4.075]], b1 = [0, −0.925],
///                 W2 = [[2, 0.8], [1, −1.3]],     b2 = [−2.95, 5.95]
///        so now y1 = [0, 6.225] and ξ2 = [2.03, −2.1425]
static void check_after_one_step(nn::Network &network) {
  network.evaluate(linalg::Vector{1.0, 1.0});
  // Softmax of two values: y0 = 1 / (1 + e^(ξ2[1] − ξ2[0]))
  const double expected = 1.0 / (1.0 + std::exp(-2.1425 - 2.03));
  CHECK_NEAR(network.outputs()[0], expected); // 0.5 before, moved toward t
  CHECK_NEAR(network.outputs()[1], 1.0 - expected);
}

/// @brief One sample, one update
static void test_learning_step() {
  nn::Network network = learning_network();
  const linalg::Vector x{1.0, 1.0};
  const linalg::Vector t{1.0, 0.0};

  network.evaluate(x);
  CHECK_NEAR(network.outputs()[0], 0.5);
  CHECK_NEAR(network.outputs()[1], 0.5);

  network.backward(x, t);
  network.update(0.1);
  check_after_one_step(network);
}

/// @brief update clears the gradients, so a second update without a backward
///        in between changes nothing
static void test_update_clears_gradients() {
  nn::Network network = learning_network();
  const linalg::Vector x{1.0, 1.0};
  const linalg::Vector t{1.0, 0.0};

  network.evaluate(x);
  network.backward(x, t);
  network.update(0.1);
  network.update(0.1);
  check_after_one_step(network);
}

int main() {
  RUN(test_evaluate_passes_outputs);
  RUN(test_evaluate_xor);
  RUN(test_evaluate_large_network);
  RUN(test_learning_step);
  RUN(test_update_clears_gradients);
  return test::summary();
}
