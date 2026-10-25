# Unit tests, added by the root CMakeLists.txt only in Debug builds.
# Source paths are relative to this folder; headers are included from src/.

add_executable(vector_tests vector_tests.cpp)
target_include_directories(vector_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME vector_tests COMMAND vector_tests)

add_executable(matrix_tests matrix_tests.cpp)
target_include_directories(matrix_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME matrix_tests COMMAND matrix_tests)

add_executable(matrix_vector_tests matrix_vector_tests.cpp)
target_include_directories(matrix_vector_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME matrix_vector_tests COMMAND matrix_vector_tests)

# Layer is not header-only, so its .cpp files are compiled into the test
add_executable(layer_tests
    layer_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/neural_network/layer.cpp
    ${CMAKE_SOURCE_DIR}/src/neural_network/activation_functions.cpp
)
target_include_directories(layer_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME layer_tests COMMAND layer_tests)
