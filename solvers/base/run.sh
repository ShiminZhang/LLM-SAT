#!/bin/bash

# Exit on any error
set -e

# Check if we have the right number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_cnf_file> <proof_output_directory>"
    exit 1
fi

# Get the absolute paths of input parameters
CNF_FILE=$(realpath "$1")
PROOF_DIR=$(realpath "$2")
PROOF_FILE="$PROOF_DIR/proof.out"

# Check if the CNF file exists
if [ ! -f "$CNF_FILE" ]; then
    echo "Error: CNF file $CNF_FILE does not exist"
    exit 1
fi

# Check if the proof directory exists
if [ ! -d "$PROOF_DIR" ]; then
    echo "Error: Proof directory $PROOF_DIR does not exist"
    exit 1
fi

# Print information about execution
echo "Running kissat on $CNF_FILE"
echo "Proof will be written to $PROOF_FILE"

# Run kissat with the specified parameters
# Note: --no-binary ensures the proof is in ASCII format
./build/kissat  --coldrestart=1 --coldrestartfc=1 --coldrestartfo=0 --coldrestartfp=0   "$CNF_FILE" "$PROOF_FILE" --no-binary

# Check if the solver ran successfully
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: kissat solver exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Solver execution completed"
