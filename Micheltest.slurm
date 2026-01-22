#!/bin/bash
#SBATCH --job-name=Myjob
#SBATCH --output=michel_output.txt
#SBATCH --error=michel_error.txt
#SBATCH --time=1000:00:00
#SBATCH --partition=longjobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=30G

# Define base path for input files
BASE_PATH="/data41/coherent/data/d2o/processedData_H2O"

# Load necessary modules (if needed)
# module load gcc/11.2.0
# module load root/6.28.04

# Log job details
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Current working directory: $(pwd)"
echo "Starting job at: $(date)"
echo "ROOT version: $(root-config --version 2>/dev/null || echo 'ROOT not in PATH')"

# Verify executable exists
EXECUTABLE="Micheltest"
if [ ! -f "./$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE executable not found in $(pwd)"
    echo "Files in directory:"
    ls -la
    echo ""
    echo "Try compiling with: g++ withMichelFitAndStoreResultsinRootfileMODULE2.cpp \$(root-config --cflags) -o $EXECUTABLE \$(root-config --libs | sed 's/-lROOTNTupleUtil//g')"
    exit 1
fi

# Check if executable is actually executable
if [ ! -x "./$EXECUTABLE" ]; then
    echo "Error: $EXECUTABLE is not executable"
    chmod +x "./$EXECUTABLE"
    echo "Made $EXECUTABLE executable"
fi

# Define input files
input_files=()
# Add all files in the desired order: 4598 first, then 4598-4608
for run in 4598 {4598..4608}; do
    file="${BASE_PATH}/run${run}_processed_H2O_v5.root"
    if [ -f "$file" ]; then
        input_files+=("$file")
        echo "Found: $file"
    else
        echo "Warning: File $file not found, skipping"
    fi
done

if [ ${#input_files[@]} -lt 2 ]; then
    echo "Error: Insufficient input files found (need at least calibration and one data file)"
    echo "Found files:"
    for file in "${input_files[@]}"; do
        echo "  $file"
    done
    exit 1
fi

# Print input files
echo "Input files to process:"
echo "  Calibration file: ${input_files[0]}"
echo "  Data files:"
for ((i=1; i<${#input_files[@]}; i++)); do
    echo "    ${input_files[$i]}"
done

# Run the analysis (first file is calibration, rest are data files)
echo "Executing: ./$EXECUTABLE ${input_files[@]}"
echo "Starting analysis at: $(date)"

# Time the execution
start_time=$(date +%s)
./$EXECUTABLE "${input_files[@]}"
exit_code=$?
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "Analysis finished at: $(date)"
echo "Runtime: $runtime seconds"

# Check if the program ran successfully
if [ $exit_code -ne 0 ]; then
    echo "Error: $EXECUTABLE failed with exit code $exit_code"
    exit $exit_code
fi

# Verify output directory (find the latest AnalysisOutput_* directory)
output_dir=$(ls -td ./AnalysisOutput_* 2>/dev/null | head -n 1)
if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
    echo "Output saved in $(pwd)/$output_dir"
    echo "Contents:"
    ls -l "$output_dir/"
    # Show summary of root files if any
    find "$output_dir" -name "*.root" | head -5 | while read rootfile; do
        echo "ROOT file: $rootfile"
    done
else
    echo "Warning: No AnalysisOutput_* directory found"
    echo "Checking for other output files:"
    find . -maxdepth 1 -type f -name "*.root" -o -name "*.txt" -o -name "*.log" | head -10
fi

echo "Job finished at: $(date)"
