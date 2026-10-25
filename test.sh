#!/bin/bash
## Builds the Debug configuration (the only one with tests) and runs all tests.
## Extra arguments are passed to ctest, e.g. `./test.sh -R vector` runs only
## tests whose name contains "vector".

set -e

# Run from the project root, wherever the script is called from
cd "$(dirname "$0")"

echo "#################"
echo "    COMPILING    "
echo "#################"

cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug


echo "#################"
echo "     TESTING     "
echo "#################"

ctest --test-dir build-debug --output-on-failure "$@"
