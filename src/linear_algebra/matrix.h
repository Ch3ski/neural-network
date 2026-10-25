#pragma once

#include <vector>

namespace linalg
{

/// Dense matrix of doubles stored row-major in one contiguous block:
/// row 0 first, then row 1, and so on.
/// Used for weight matrices W and their gradients dW.
class Matrix
{
public:

private:
    /// The elements, row after row.
    std::vector<double> m_data;
};

} // namespace linalg
