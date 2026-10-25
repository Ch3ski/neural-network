#include "linear_algebra/matrix.h"
#include "test_utils.h"

#include <cstddef>

/// @brief Matrix(rows, cols, value) has the given shape and every element equal
/// to value
static void test_size_value_constructor() {
  linalg::Matrix m(2, 3, 0.5);
  CHECK(m.rows() == 2);
  CHECK(m.cols() == 3);
  CHECK(m.size() == 6);
  for (std::size_t row = 0; row < m.rows(); ++row) {
    for (std::size_t col = 0; col < m.cols(); ++col) {
      CHECK_NEAR(m(row, col), 0.5);
    }
  }
}

/// @brief A matrix with 0 rows has no elements
static void test_size_value_constructor_zero_rows() {
  linalg::Matrix m(0, 5, 1.0);
  CHECK(m.rows() == 0);
  CHECK(m.cols() == 5);
  CHECK(m.size() == 0);
}

/// @brief Matrix{{...}, {...}}: outer braces are rows, inner braces are the
/// values of one row
static void test_list_constructor() {
  linalg::Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  CHECK(m.rows() == 2);
  CHECK(m.cols() == 3);
  CHECK(m.size() == 6);
  CHECK_NEAR(m(0, 0), 1.0);
  CHECK_NEAR(m(0, 1), 2.0);
  CHECK_NEAR(m(0, 2), 3.0);
  CHECK_NEAR(m(1, 0), 4.0);
  CHECK_NEAR(m(1, 1), 5.0);
  CHECK_NEAR(m(1, 2), 6.0);
}

/// @brief One row gives a 1 × n matrix, one value per row gives an n × 1 matrix
static void test_list_constructor_single_row_and_column() {
  linalg::Matrix row{{1.0, 2.0, 3.0}};
  CHECK(row.rows() == 1);
  CHECK(row.cols() == 3);
  CHECK_NEAR(row(0, 2), 3.0);

  linalg::Matrix column{{1.0}, {2.0}, {3.0}};
  CHECK(column.rows() == 3);
  CHECK(column.cols() == 1);
  CHECK_NEAR(column(2, 0), 3.0);

  linalg::Matrix single{{7.0}};
  CHECK(single.rows() == 1);
  CHECK(single.cols() == 1);
  CHECK_NEAR(single(0, 0), 7.0);
}

/// @brief Empty braces give a 0 × 0 matrix
static void test_list_constructor_empty() {
  linalg::Matrix m{};
  CHECK(m.rows() == 0);
  CHECK(m.cols() == 0);
  CHECK(m.size() == 0);
}

/// @brief Weight matrix orientation: W(j, i) is the weight from input i to
/// neuron j
static void test_weight_matrix_orientation() {
  // W₁ from the hand-computed example: 2 neurons (rows) × 2 inputs (columns)
  linalg::Matrix w{{0.5, -1.0}, {1.0, 0.5}};
  CHECK_NEAR(w(0, 0), 0.5);  // neuron 0, input x₁
  CHECK_NEAR(w(0, 1), -1.0); // neuron 0, input x₂
  CHECK_NEAR(w(1, 0), 1.0);  // neuron 1, input x₁
  CHECK_NEAR(w(1, 1), 0.5);  // neuron 1, input x₂
}

/// @brief Writing one element changes exactly that element
static void test_write_element() {
  linalg::Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  m(1, 0) = 7.0;
  CHECK_NEAR(m(1, 0), 7.0);
  // neighbours in memory and in the same column stay unchanged
  CHECK_NEAR(m(0, 2), 3.0);
  CHECK_NEAR(m(1, 1), 5.0);
  CHECK_NEAR(m(0, 0), 1.0);
}

/// @brief Every (row, col) of a non-square matrix is a separate element,
///        which fails if the index formula mixes up rows and columns
static void test_row_major_indexing() {
  linalg::Matrix m(3, 4, 0.0);
  for (std::size_t row = 0; row < m.rows(); ++row) {
    for (std::size_t col = 0; col < m.cols(); ++col) {
      m(row, col) = 10.0 * static_cast<double>(row) + static_cast<double>(col);
    }
  }
  for (std::size_t row = 0; row < m.rows(); ++row) {
    for (std::size_t col = 0; col < m.cols(); ++col) {
      CHECK_NEAR(m(row, col),
                 10.0 * static_cast<double>(row) + static_cast<double>(col));
    }
  }
}

/// @brief The const operator() reads the same data as the non-const one
static void test_const_access() {
  linalg::Matrix m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  const linalg::Matrix &view = m;
  CHECK_NEAR(view(1, 2), 6.0);

  m(1, 2) = -1.0;
  CHECK_NEAR(view(1, 2), -1.0);
}

/// @brief A copy is independent: changing it does not change the original
static void test_copy_is_independent() {
  linalg::Matrix original{{1.0, 2.0}, {3.0, 4.0}};
  linalg::Matrix copy = original;
  copy(0, 0) = 100.0;
  CHECK_NEAR(copy(0, 0), 100.0);
  CHECK_NEAR(original(0, 0), 1.0);
  CHECK(copy.rows() == original.rows());
  CHECK(copy.cols() == original.cols());
}

/// @brief Writing and reading every element of a hidden-layer weight matrix
///        (128 × 784): each element keeps its own value
static void test_large_row_major_indexing() {
  linalg::Matrix m(128, 784, 0.0);

  test::benchmark("Matrix write all elements (128 x 784)", 100, [&] {
    for (std::size_t row = 0; row < m.rows(); ++row) {
      for (std::size_t col = 0; col < m.cols(); ++col) {
        m(row, col) =
            1000.0 * static_cast<double>(row) + static_cast<double>(col);
      }
    }
  });

  std::size_t wrong = 0;
  for (std::size_t row = 0; row < m.rows(); ++row) {
    for (std::size_t col = 0; col < m.cols(); ++col) {
      if (m(row, col) !=
          1000.0 * static_cast<double>(row) + static_cast<double>(col)) {
        ++wrong;
      }
    }
  }
  CHECK(wrong == 0);
}

int main() {
  RUN(test_size_value_constructor);
  RUN(test_size_value_constructor_zero_rows);
  RUN(test_list_constructor);
  RUN(test_list_constructor_single_row_and_column);
  RUN(test_list_constructor_empty);
  RUN(test_weight_matrix_orientation);
  RUN(test_write_element);
  RUN(test_row_major_indexing);
  RUN(test_const_access);
  RUN(test_copy_is_independent);
  RUN(test_large_row_major_indexing);
  return test::summary();
}
