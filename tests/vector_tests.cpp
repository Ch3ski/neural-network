#include "linear_algebra/vector.h"
#include "test_utils.h"

#include <algorithm>
#include <cstddef>

/// @brief Vector(size, value) creates size elements, all equal to value
static void test_size_value_constructor() {
  linalg::Vector v(3, 0.5);
  CHECK(v.size() == 3);
  CHECK_NEAR(v[0], 0.5);
  CHECK_NEAR(v[1], 0.5);
  CHECK_NEAR(v[2], 0.5);
}

/// @brief += and -= on a vector the size of one image (784): adding and
///        subtracting the same vector gives back the original values
static void test_large_add_subtract() {
  linalg::Vector v(784, 1.5);
  linalg::Vector other(784, 0.25);

  test::benchmark("Vector += and -= (784)", 10000, [&] {
    v += other;
    v -= other;
  });

  std::size_t wrong = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] != 1.5) {
      ++wrong;
    }
  }
  CHECK(wrong == 0);
}

/// @brief apply(source, f) on a hidden layer (128): ReLU of source[i] = i − 64
///        gives max(0, i − 64), and source stays unchanged
static void test_large_apply_from_source() {
  linalg::Vector source(128, 0.0);
  for (std::size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<double>(i) - 64.0;
  }
  linalg::Vector out(128, 0.0);

  test::benchmark("Vector apply from source (128)", 10000, [&] {
    out.apply(source, [](double x) { return std::max(0.0, x); });
  });

  std::size_t wrong = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double x = static_cast<double>(i) - 64.0;
    if (out[i] != std::max(0.0, x) || source[i] != x) {
      ++wrong;
    }
  }
  CHECK(wrong == 0);
}

int main() {
  RUN(test_size_value_constructor);
  RUN(test_large_add_subtract);
  RUN(test_large_apply_from_source);
  return test::summary();
}
