#pragma once

#include "linear_algebra/matrix.h"
#include "linear_algebra/vector.h"

namespace nn
{

class Layer
{
public:

private:
    linalg::Matrix m_weights;
    linalg::Vector m_bias;
};

} // namespace nn
