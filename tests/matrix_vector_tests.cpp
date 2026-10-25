#include "linear_algebra/matrix_vector.h"
#include "test_utils.h"

#include <cstddef>

/// @brief A 2 × 3 matrix times a 3-vector gives a 2-vector:
///        out[row] = Σ matrix(row, column) · vector[column]
static void test_multiply_wide() {
  linalg::Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  linalg::Vector v{1.0, 2.0, 3.0};
  linalg::Vector out(2, 0.0);
  linalg::Multiply(m, v, out);
  CHECK_NEAR(out[0], 14.0); // 1·1 + 2·2 + 3·3
  CHECK_NEAR(out[1], 32.0); // 4·1 + 5·2 + 6·3
}

/// @brief A 3 × 2 matrix times a 2-vector gives a 3-vector, which fails if
///        the loops mix up rows and columns
static void test_multiply_tall() {
  linalg::Matrix m{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  linalg::Vector v{2.0, 1.0};
  linalg::Vector out(3, 0.0);
  linalg::Multiply(m, v, out);
  CHECK_NEAR(out[0], 4.0);  // 1·2 + 2·1
  CHECK_NEAR(out[1], 10.0); // 3·2 + 4·1
  CHECK_NEAR(out[2], 16.0); // 5·2 + 6·1
}

/// @brief Inner potentials ξ = W · x of a layer with 2 neurons and 2 inputs,
///        with negative and fractional weights
static void test_multiply_weight_matrix() {
  linalg::Matrix w{{0.5, -1.0}, {1.0, 0.5}};
  linalg::Vector x{1.0, 2.0};
  linalg::Vector xi(2, 0.0);
  linalg::Multiply(w, x, xi);
  CHECK_NEAR(xi[0], -1.5); // 0.5·1 − 1.0·2
  CHECK_NEAR(xi[1], 2.0);  // 1.0·1 + 0.5·2
}

/// @brief Multiplying by the unit vector e_k gives column k of the matrix,
///        so vector[k] is multiplied with column k and not with row k
static void test_multiply_unit_vector_picks_column() {
  linalg::Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  linalg::Vector e1{0.0, 1.0, 0.0};
  linalg::Vector out(2, 0.0);
  linalg::Multiply(m, e1, out);
  CHECK_NEAR(out[0], 2.0);
  CHECK_NEAR(out[1], 5.0);
}

/// @brief Old values in out are replaced, not added to: calling Multiply
///        again with the same out gives the same result
static void test_multiply_overwrites_out() {
  linalg::Matrix m{{1.0, 2.0}, {3.0, 4.0}};
  linalg::Vector v{1.0, 1.0};
  linalg::Vector out(2, 99.0);

  linalg::Multiply(m, v, out);
  CHECK_NEAR(out[0], 3.0);
  CHECK_NEAR(out[1], 7.0);

  linalg::Multiply(m, v, out);
  CHECK_NEAR(out[0], 3.0);
  CHECK_NEAR(out[1], 7.0);
}

/// @brief A 1 × n matrix gives the dot product, an n × 1 matrix scales its
///        column by the single element of vector
static void test_multiply_single_row_and_column() {
  linalg::Matrix row{{1.0, 2.0, 3.0}};
  linalg::Vector v{4.0, 5.0, 6.0};
  linalg::Vector dot(1, 0.0);
  linalg::Multiply(row, v, dot);
  CHECK_NEAR(dot[0], 32.0); // 1·4 + 2·5 + 3·6

  linalg::Matrix column{{1.0}, {2.0}, {3.0}};
  linalg::Vector scalar{2.0};
  linalg::Vector scaled(3, 0.0);
  linalg::Multiply(column, scalar, scaled);
  CHECK_NEAR(scaled[0], 2.0);
  CHECK_NEAR(scaled[1], 4.0);
  CHECK_NEAR(scaled[2], 6.0);
}

/// @brief A matrix with 0 columns gives empty sums, so every element of out
///        is reset to 0 even though the inner loop never runs
static void test_multiply_no_columns() {
  linalg::Matrix m(2, 0, 0.0);
  linalg::Vector v(0, 0.0);
  linalg::Vector out(2, 5.0);
  linalg::Multiply(m, v, out);
  CHECK_NEAR(out[0], 0.0);
  CHECK_NEAR(out[1], 0.0);
}

/// @brief Hidden layer size, 128 × 784: W(row, column) = row + 1 and
///        x[column] = column give out[row] = (row + 1) · (0 + 1 + … + 783)
static void test_multiply_large_hidden_layer() {
  linalg::Matrix m(128, 784, 0.0);
  for (std::size_t row = 0; row < m.rows(); ++row) {
    for (std::size_t column = 0; column < m.cols(); ++column) {
      m(row, column) = static_cast<double>(row + 1);
    }
  }
  linalg::Vector v(784, 0.0);
  for (std::size_t column = 0; column < v.size(); ++column) {
    v[column] = static_cast<double>(column);
  }
  linalg::Vector out(128, 0.0);

  test::benchmark("Multiply 128 x 784", 200,
                  [&] { linalg::Multiply(m, v, out); });

  const double columnSum = 783.0 * 784.0 / 2.0; // 0 + 1 + … + 783
  std::size_t wrong = 0;
  for (std::size_t row = 0; row < out.size(); ++row) {
    if (out[row] != static_cast<double>(row + 1) * columnSum) {
      ++wrong;
    }
  }
  CHECK(wrong == 0);
}

/// @brief Output layer size, 10 × 128: W(row, column) = row + 1 and x = all 1
///        give out[row] = 128 · (row + 1)
static void test_multiply_large_output_layer() {
  linalg::Matrix m(10, 128, 0.0);
  for (std::size_t row = 0; row < m.rows(); ++row) {
    for (std::size_t column = 0; column < m.cols(); ++column) {
      m(row, column) = static_cast<double>(row + 1);
    }
  }
  linalg::Vector v(128, 1.0);
  linalg::Vector out(10, 0.0);

  test::benchmark("Multiply 10 x 128", 2000,
                  [&] { linalg::Multiply(m, v, out); });

  for (std::size_t row = 0; row < out.size(); ++row) {
    CHECK_NEAR(out[row], 128.0 * static_cast<double>(row + 1));
  }
}

int main() {
  RUN(test_multiply_wide);
  RUN(test_multiply_tall);
  RUN(test_multiply_weight_matrix);
  RUN(test_multiply_unit_vector_picks_column);
  RUN(test_multiply_overwrites_out);
  RUN(test_multiply_single_row_and_column);
  RUN(test_multiply_no_columns);
  RUN(test_multiply_large_hidden_layer);
  RUN(test_multiply_large_output_layer);
  return test::summary();
}
