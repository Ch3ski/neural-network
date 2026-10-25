#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace linalg
{

/// Dense matrix of doubles stored row-major in one contiguous block:
/// row 0 first, then row 1, and so on.
/// Used for weight matrices W and their gradients dW.

class Matrix
{
public:
    using RowList = std::initializer_list<double>;

    explicit Matrix(std::size_t rows, std::size_t cols, double value);

    explicit Matrix(std::initializer_list<RowList> data);

    std::size_t size() const;
    std::size_t rows() const;
    std::size_t cols() const;

    double &operator()(std::size_t row, std::size_t column);
    const double &operator()(std::size_t row, std::size_t column) const;

private:
    std::size_t m_rows;
    std::size_t m_cols;
    std::vector<double> m_data;
};

inline Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : m_rows{rows}
    , m_cols{cols}
    , m_data(rows * cols, value)
{
}

inline Matrix::Matrix(std::initializer_list<RowList> data)
    : m_rows{data.size()}
    , m_cols{data.size() == 0 ? 0 : data.begin()->size()}
{
    // Row-major: append each row's values after the previous row
    m_data.reserve(m_rows * m_cols);
    for (const RowList &row : data) {
        assert(row.size() == m_cols && "Matrix: all rows must have the same length");

        m_data.insert(m_data.end(), row.begin(), row.end());
    }
}

inline std::size_t Matrix::size() const
{
    return m_data.size();
}

inline std::size_t Matrix::rows() const
{
    return m_rows;
}

inline std::size_t Matrix::cols() const
{
    return m_cols;
}

inline double &Matrix::operator()(std::size_t row, std::size_t column)
{
    // Row-major: skip row full rows, then move column elements into the row
    return m_data[row * m_cols + column];
}

inline const double &Matrix::operator()(std::size_t row, std::size_t column) const
{
    return m_data[row * m_cols + column];
}

} // namespace linalg
