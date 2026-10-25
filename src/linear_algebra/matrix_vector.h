#pragma once

#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"

#include <cassert>
#include <cstddef>

/// Operations that combine a Matrix with a Vector.
/// Results are written into a preallocated output vector instead of being returned.
namespace linalg
{

/// @brief Multiplies matrix by vector, out = matrix · vector
/// @param matrix Matrix W, size rows × cols
/// @param vector Vector x, size cols
/// @param out    Vector the result is written into, size rows
/// @pre matrix.cols() == vector.size() and matrix.rows() == out.size()
inline void Multiply(const Matrix &matrix, const Vector &vector, Vector &out)
{
    assert(matrix.cols() == vector.size() && "Multiply: matrix columns differ from vector size");
    assert(matrix.rows() == out.size() && "Multiply: matrix rows differ from out size");
    assert(&vector != &out && "Multiply: out must not be the input vector");

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        out[row] = 0.0;
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            out[row] += matrix(row, column) * vector[column];
        }
    }
}

} // namespace linalg
