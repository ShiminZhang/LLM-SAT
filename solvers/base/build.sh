#!/bin/bash

# Exit on any error
set -e

# Script for building kissat solver
echo "Starting kissat build process..."

# Change to the kissat directory if needed
# Uncomment the next line if the script is not executed from the kissat directory
# cd kissat

# Check and fix executable permissions for configure script
if [ ! -x ./configure ]; then
  echo "Configure script is not executable. Setting execute permission..."
  chmod +x ./configure
fi

# Check and fix executable permissions for any *.sh files
echo "Setting execute permission for all shell scripts..."
find . -name "*.sh" -type f -exec chmod +x {} \;

# Check and fix executable permissions for scripts in scripts directory if it exists
if [ -d "./scripts" ]; then
  echo "Setting execute permission for all scripts in scripts directory..."
  find ./scripts -type f -exec chmod +x {} \;
fi

# Run configure
echo "Running configure..."
./configure

# Run make to build kissat
echo "Running make..."
make

# Check if build was successful
if [ -f "./build/kissat" ]; then
  echo "Build successful! Kissat binary created."
  ls -l ./build/kissat
else
  echo "Build failed. kissat binary not found."
  exit 1
fi

echo "Build process completed."
