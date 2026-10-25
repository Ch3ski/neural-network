#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace linalg
{

/// A callable that can be called with a double and returns exactly a double,
/// e.g. [](double x) { return x * 2.0; } or a function double f(double).
template <typename F>
concept DoubleFunction = std::invocable<F, double> && std::same_as<std::invoke_result_t<F, double>, double>;

/// Used for biases b, inner potentials ξ, outputs y and gradients δ.
class Vector
{
public:
    /// Returns the number of elements.
    std::size_t size() const;

    /// Returns element i for reading or writing.
    /// No bounds check: i must be smaller than size().
    double &operator[](std::size_t i);
    /// Returns element i of a const vector, read-only.
    /// No bounds check: i must be smaller than size().
    const double &operator[](std::size_t i) const;

    /// Replaces every element x with function(x), in place.
    /// function takes a double and returns a double (e.g. a lambda);
    /// anything else is rejected at compile time by DoubleFunction.
    template <DoubleFunction Function>
    void apply(Function function);

    /// Multiplies every element by scalar, in place.
    Vector &operator*=(double scalar);
    /// Divides every element by scalar, in place.
    Vector &operator/=(double scalar);

private:
    std::vector<double> m_data;
};

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

template <DoubleFunction F>
void Vector::apply(F function)
{
    for (double &value : m_data) {
        value = function(value);
    }
}

inline Vector &Vector::operator*=(double scalar)
{
    for (double &value : m_data) {
        value *= scalar;
    }
    return *this;
}

inline Vector &Vector::operator/=(double scalar)
{
    for (double &value : m_data) {
        value /= scalar;
    }
    return *this;
}

// Copy versions: the parameter v is taken by value, so it already is the copy.
// It is modified and returned (moved out, no second copy).

/// Returns a copy of v with every element multiplied by scalar (v * scalar).
inline Vector operator*(Vector v, double scalar)
{
    v *= scalar;
    return v;
}

/// Returns a copy of v with every element multiplied by scalar (scalar * v).
inline Vector operator*(double scalar, Vector v)
{
    v *= scalar;
    return v;
}

/// Returns a copy of v with every element divided by scalar.
inline Vector operator/(Vector v, double scalar)
{
    v /= scalar;
    return v;
}

} // namespace linalg
