#include "linear_algebra/vector.h"
#include "test_utils.h"

/// @brief Vector(size, value) creates size elements, all equal to value
static void test_size_value_constructor() {
  linalg::Vector v(3, 0.5);
  CHECK(v.size() == 3);
  CHECK_NEAR(v[0], 0.5);
  CHECK_NEAR(v[1], 0.5);
  CHECK_NEAR(v[2], 0.5);
}

int main() {
  RUN(test_size_value_constructor);
  return test::summary();
}
