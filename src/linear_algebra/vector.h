#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace linalg
{

/// @brief A callable that takes a double and returns exactly a double,
///        e.g. [](double x) { return x * 2.0; } or a function double f(double)
/// @tparam F Type of the callable
template <typename F>
concept DoubleFunction = std::invocable<F, double> && std::same_as<std::invoke_result_t<F, double>, double>;

/// @brief Dense vector of doubles stored contiguously
///        Used for biases b, inner potentials ξ, outputs y and gradients δ
class Vector
{
public:
    /// @brief Creates a vector of size elements, all set to value
    /// @param size  Number of elements
    /// @param value Initial value of every element
    explicit Vector(std::size_t size, double value);

    /// @brief Creates a vector from the listed values
    /// @param list Values of the elements, in order
    explicit Vector(std::initializer_list<double> data);

    /// @brief Returns the number of elements
    /// @return Number of elements
    std::size_t size() const;

    /// @brief Accesses element i for reading or writing
    /// @param i Index of the element
    /// @return Reference to element i
    /// @warning No bounds check: i must be smaller than size()
    double &operator[](std::size_t i);

    /// @brief Accesses element i of a const vector, read-only
    /// @param i Index of the element
    /// @return Const reference to element i
    /// @warning No bounds check: i must be smaller than size()
    const double &operator[](std::size_t i) const;

    /// @brief Replaces every element x with function(x), in place
    /// @tparam Function Callable taking and returning a double, checked by DoubleFunction
    /// @param function Function applied to every element, e.g. a lambda
    template <DoubleFunction Function>
    void apply(Function function);

    /// @brief Sets every element i to function(source[i]), source is unchanged
    /// @tparam Function Callable taking and returning a double, checked by DoubleFunction
    /// @param source   Vector to read from
    /// @param function Function applied to every element of source, e.g. a lambda
    /// @pre source.size() == size()
    template <DoubleFunction Function>
    void apply(const Vector &source, Function function);

    /* OPERATIONS */

    /// @brief Adds other element by element, in place
    /// @param other Vector to add
    /// @return Reference to this vector
    /// @pre other.size() == size()
    Vector &operator+=(const Vector &other);

    /// @brief Subtracts other element by element, in place
    /// @param other Vector to subtract
    /// @return Reference to this vector
    /// @pre other.size() == size()
    Vector &operator-=(const Vector &other);

    /// @brief Multiplies every element by scalar, in place
    /// @param scalar Factor
    /// @return Reference to this vector
    Vector &operator*=(double scalar);

    /// @brief Divides every element by scalar, in place
    /// @param scalar Divisor
    /// @return Reference to this vector
    Vector &operator/=(double scalar);

    /// @brief Adds scalar to every element, in place
    /// @param scalar Value to add
    /// @return Reference to this vector
    Vector &operator+=(double scalar);

    /// @brief Subtracts scalar from every element, in place
    /// @param scalar Value to subtract
    /// @return Reference to this vector
    Vector &operator-=(double scalar);

private:
    std::vector<double> m_data;
};

inline Vector::Vector(std::size_t size, double value)
    : m_data(size, value)
{
}

inline Vector::Vector(std::initializer_list<double> data)
    : m_data{data}
{
}

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

template <DoubleFunction F>
void Vector::apply(const Vector &source, F function)
{
    assert(size() == source.size() && "Vector apply: sizes differ");

    for (std::size_t i = 0; i < size(); ++i) {
        m_data[i] = function(source[i]);
    }
}

inline Vector &Vector::operator+=(const Vector &other)
{
    assert(size() == other.size() && "Vector +=: sizes differ");

    for (std::size_t i = 0; i < size(); ++i) {
        m_data[i] += other[i];
    }

    return *this;
}

inline Vector &Vector::operator-=(const Vector &other)
{
    assert(size() == other.size() && "Vector -=: sizes differ");

    for (std::size_t i = 0; i < size(); ++i) {
        m_data[i] -= other[i];
    }

    return *this;
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

inline Vector &Vector::operator+=(double scalar)
{
    for (double &value : m_data) {
        value += scalar;
    }
    return *this;
}

inline Vector &Vector::operator-=(double scalar)
{
    for (double &value : m_data) {
        value -= scalar;
    }
    return *this;
}

// Copy versions: the parameter v is taken by value, so it already is the copy
// It is modified and returned (moved out, no second copy)

/// @brief Multiplies every element by scalar, as a copy (v * scalar)
/// @param v      Vector to multiply, taken by value as the copy
/// @param scalar Factor
/// @return New vector with the results, v at the call site is unchanged
inline Vector operator*(Vector v, double scalar)
{
    v *= scalar;
    return v;
}

/// @brief Multiplies every element by scalar, as a copy (scalar * v)
/// @param scalar Factor
/// @param v      Vector to multiply, taken by value as the copy
/// @return New vector with the results, v at the call site is unchanged
inline Vector operator*(double scalar, Vector v)
{
    v *= scalar;
    return v;
}

/// @brief Divides every element by scalar, as a copy (v / scalar)
/// @param v      Vector to divide, taken by value as the copy
/// @param scalar Divisor
/// @return New vector with the results, v at the call site is unchanged
inline Vector operator/(Vector v, double scalar)
{
    v /= scalar;
    return v;
}

/// @brief Adds scalar to every element, as a copy (v + scalar)
/// @param v      Vector to add to, taken by value as the copy
/// @param scalar Value to add
/// @return New vector with the results, v at the call site is unchanged
inline Vector operator+(Vector v, double scalar)
{
    v += scalar;
    return v;
}

/// @brief Subtracts scalar from every element, as a copy (v - scalar)
/// @param v      Vector to subtract from, taken by value as the copy
/// @param scalar Value to subtract
/// @return New vector with the results, v at the call site is unchanged
inline Vector operator-(Vector v, double scalar)
{
    v -= scalar;
    return v;
}

} // namespace linalg
