#!/bin/bash

# Loop through numbers 1 to 9
for i in {1..9}; do
    # Format folder name (ensures 0 before single digits)
    folder_name="run_0$i"
    
    # Path to the file
    file_path="events/$folder_name/unweighted_events.lhe"
    
    # Check if the file exists before running the script
    if [ -f "$file_path" ]; then
        echo "Processing $file_path with number $i"
        python3 run_amp_scan_parallel.py "$file_path" "$i"
    else
        echo "Warning: File $file_path not found, skipping..."
    fi
done

