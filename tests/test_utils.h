#pragma once

#include <cmath>
#include <cstdio>

/// Minimal test helpers: no framework, just checks that count failures.
/// Use the CHECK / CHECK_NEAR / RUN macros below, and end main() with
/// `return test::summary();` so CTest sees a non-zero exit code on failure.
namespace test {

/// @brief Number of checks run so far
inline int checks = 0;
/// @brief Number of checks that failed so far
inline int failures = 0;

/// @brief Records one check and prints it if it failed
/// @param ok         Result of the check
/// @param file       Source file of the check
/// @param line       Line of the check
/// @param expression The checked condition as text
inline void report(bool ok, const char *file, int line,
                   const char *expression) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("%s:%d: FAIL %s\n", file, line, expression);
  }
}

/// @brief Records a comparison of two doubles with a tolerance, prints both
///        values if it failed
/// @param actual     Value computed by the code under test
/// @param expected   Value it should have
/// @param file       Source file of the check
/// @param line       Line of the check
/// @param expression The checked value as text
/// @param tolerance  Largest allowed difference
inline void report_near(double actual, double expected, const char *file,
                        int line, const char *expression,
                        double tolerance = 1e-9) {
  ++checks;
  if (!(std::abs(actual - expected) <= tolerance)) {
    ++failures;
    std::printf("%s:%d: FAIL %s is %.12g, expected %.12g\n", file, line,
                expression, actual, expected);
  }
}

/// @brief Prints how many checks ran and failed
/// @return Exit code for main(): 0 if every check passed, 1 otherwise
inline int summary() {
  std::printf("%d checks, %d failed\n", checks, failures);
  return failures == 0 ? 0 : 1;
}

} // namespace test

/// @brief Checks that condition is true, e.g. CHECK(v.size() == 3)
#define CHECK(condition)                                                       \
  test::report((condition), __FILE__, __LINE__, #condition)

/// @brief Checks that two doubles are equal within 1e-9, e.g. CHECK_NEAR(v[0],
/// 0.5)
#define CHECK_NEAR(actual, expected)                                           \
  test::report_near((actual), (expected), __FILE__, __LINE__, #actual)

/// @brief Runs one test function and prints its name
#define RUN(test_function)                                                     \
  do {                                                                         \
    std::printf("%s\n", #test_function);                                       \
    test_function();                                                           \
  } while (false)
