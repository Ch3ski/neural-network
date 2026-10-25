#pragma once

#include <cstddef>
#include <vector>

namespace linalg
{

class Vector
{
public:
    std::size_t size() const;

    double &operator[](std::size_t i);
    const double &operator[](std::size_t i) const;

private:
    std::vector<double> m_data;
};

// Small hot-path accessors are defined here as inline (not in vector.cpp),
// because an inline function must be visible in every .cpp that calls it.

inline std::size_t Vector::size() const
{
    return m_data.size();
}

inline double &Vector::operator[](std::size_t i)
{
    return m_data[i];
}

inline const double &Vector::operator[](std::size_t i) const
{
    return m_data[i];
}

} // namespace linalg
