#!/bin/bash
# Setup Kissat SAT solver base template
# This script downloads and configures the Kissat solver that will be used as a template
# for generating modified solvers with LLM-generated restart policies.

set -euo pipefail

echo "Setting up Kissat solver base template..."

# Create solvers directory if it doesn't exist
mkdir -p solvers

# Remove old base if it exists
if [ -d "solvers/base" ]; then
    echo "Removing old solvers/base directory..."
    rm -rf solvers/base
fi

# Clone Kissat repository (official version)
echo "Cloning Kissat repository..."
git clone https://github.com/arminbiere/kissat.git solvers/base

cd solvers/base

# Check out a stable version (adjust tag as needed)
# You can find available tags with: git tag -l
echo "Checking out stable version..."
git checkout rel-3.1.1 || echo "Warning: Could not checkout rel-3.1.1, using default branch"

# Run configure to set up the build system
echo "Configuring Kissat..."
./configure

# Optional: Do an initial build to verify it works
echo "Building Kissat (test build)..."
make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "✓ Kissat solver base template setup complete!"
echo ""
echo "Next steps:"
echo "1. Check that solvers/base/src/restart.c exists"
echo "2. Ensure it has //LLMSAT start and //LLMSAT end markers"
echo "3. If markers are missing, you need to add them manually"
echo ""
echo "To verify:"
echo "  ls -lh solvers/base/src/restart.c"
echo "  grep -n 'LLMSAT' solvers/base/src/restart.c"