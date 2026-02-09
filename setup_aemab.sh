#!/bin/bash
# Setup AE_kissatMAB solver

set -euo pipefail
tar -xf AE_kissat2025_MAB.tar.xz -C ~/scratch/LLM-SAT/solvers/
mv solvers/AE_kissat2025_MAB/* solvers/base/
rm -rf solvers/AE_kissat2025_MAB
cd solvers/base
rm makefile
cp ~/scratch/LLM-SAT/function_registry.yaml ~/scratch/LLM-SAT/solvers/base/
./configure
