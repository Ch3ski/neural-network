#include "neural_network/layer.h"
#include "test_utils.h"

#include <cmath>

using actifunc::Type;

/// @brief The constructor sizes potentials and outputs to the number of
///        neurons (rows of W), all 0 before the first evaluate
static void test_constructor_sizes() {
  nn::Layer layer(Type::ReLu, linalg::Matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                  linalg::Vector{0.0, 0.0});
  CHECK(layer.potentials().size() == 2);
  CHECK(layer.outputs().size() == 2);
  CHECK_NEAR(layer.potentials()[0], 0.0);
  CHECK_NEAR(layer.outputs()[1], 0.0);
}

/// @brief ξ = W · x + b, y = ReLU(ξ), and ξ is kept unchanged by the
///        activation
static void test_evaluate_relu() {
  nn::Layer layer(Type::ReLu, linalg::Matrix{{1.0, -2.0}, {3.0, 4.0}},
                  linalg::Vector{0.0, -1.0});
  layer.evaluate(linalg::Vector{1.0, 1.0});
  CHECK_NEAR(layer.potentials()[0], -1.0); // 1·1 − 2·1 + 0
  CHECK_NEAR(layer.potentials()[1], 6.0);  // 3·1 + 4·1 − 1
  CHECK_NEAR(layer.outputs()[0], 0.0);     // ReLU cuts the negative ξ
  CHECK_NEAR(layer.outputs()[1], 6.0);
}

/// @brief 3 inputs and 2 neurons, which fails if W is used with rows and
///        columns swapped
static void test_evaluate_non_square() {
  nn::Layer layer(Type::ReLu, linalg::Matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                  linalg::Vector{1.0, -1.0});
  layer.evaluate(linalg::Vector{1.0, 2.0, 3.0});
  CHECK_NEAR(layer.potentials()[0], 15.0); // 1 + 4 + 9 + 1
  CHECK_NEAR(layer.potentials()[1], 31.0); // 4 + 10 + 18 − 1
  CHECK_NEAR(layer.outputs()[0], 15.0);
  CHECK_NEAR(layer.outputs()[1], 31.0);
}

/// @brief Sigmoid and tanh layers apply their own activation
static void test_evaluate_sigmoid_and_tanh() {
  nn::Layer sigmoid(Type::Sigmoid, linalg::Matrix{{1.0, 1.0}},
                    linalg::Vector{0.0});
  sigmoid.evaluate(linalg::Vector{0.0, 0.0});
  CHECK_NEAR(sigmoid.outputs()[0], 0.5); // σ(0)

  nn::Layer tanh(Type::Tanh, linalg::Matrix{{2.0}}, linalg::Vector{-1.0});
  tanh.evaluate(linalg::Vector{1.0});
  CHECK_NEAR(tanh.potentials()[0], 1.0); // 2·1 − 1
  CHECK_NEAR(tanh.outputs()[0], std::tanh(1.0));
}

/// @brief Leaky ReLU scales a negative ξ by the default α = 0.01
static void test_evaluate_leaky_relu() {
  nn::Layer layer(Type::LeakyReLu, linalg::Matrix{{1.0}, {1.0}},
                  linalg::Vector{0.0, 4.0});
  layer.evaluate(linalg::Vector{-2.0});
  CHECK_NEAR(layer.outputs()[0], -0.02); // 0.01 · (−2)
  CHECK_NEAR(layer.outputs()[1], 2.0);   // −2 + 4
}

/// @brief Softmax turns ξ = [0, ln 3] into [1/4, 3/4], which sums to 1
static void test_evaluate_softmax() {
  nn::Layer layer(Type::Softmax, linalg::Matrix{{0.0}, {0.0}},
                  linalg::Vector{0.0, std::log(3.0)});
  layer.evaluate(linalg::Vector{1.0});
  CHECK_NEAR(layer.outputs()[0], 0.25); // 1 / (1 + 3)
  CHECK_NEAR(layer.outputs()[1], 0.75); // 3 / (1 + 3)
  CHECK_NEAR(layer.outputs()[0] + layer.outputs()[1], 1.0);
}

/// @brief A second evaluate overwrites the first: the result depends only on
///        the new input
static void test_evaluate_twice() {
  nn::Layer layer(Type::ReLu, linalg::Matrix{{1.0, -2.0}, {3.0, 4.0}},
                  linalg::Vector{0.0, -1.0});
  layer.evaluate(linalg::Vector{1.0, 1.0});
  layer.evaluate(linalg::Vector{2.0, 0.0});
  CHECK_NEAR(layer.potentials()[0], 2.0); // 1·2 − 2·0 + 0
  CHECK_NEAR(layer.potentials()[1], 5.0); // 3·2 + 4·0 − 1
  CHECK_NEAR(layer.outputs()[0], 2.0);
  CHECK_NEAR(layer.outputs()[1], 5.0);
}

int main() {
  RUN(test_constructor_sizes);
  RUN(test_evaluate_relu);
  RUN(test_evaluate_non_square);
  RUN(test_evaluate_sigmoid_and_tanh);
  RUN(test_evaluate_leaky_relu);
  RUN(test_evaluate_softmax);
  RUN(test_evaluate_twice);
  return test::summary();
}
