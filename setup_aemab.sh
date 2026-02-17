#!/bin/bash
# Setup AE_kissatMAB solver

set -euo pipefail
tar -xf AE_kissat2025_MAB.tar.xz -C solvers/
mv solvers/AE_kissat2025_MAB/* solvers/base/
rm -rf solvers/AE_kissat2025_MAB
cd solvers/base
./configure
