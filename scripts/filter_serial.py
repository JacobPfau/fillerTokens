import numpy as np
import glob
import os
import sys
from src.utils import get_files_from_name

def count_spaces_per_line(file_path):
    space_counts = []
    with open(file_path, 'r') as file:
        for line in file:
            space_counts.append(line.count(' '))
    return space_counts

def filter_file_by_space_threshold(source_file_path, target_file_path, threshold):
    with open(target_file_path, 'r') as target_file:
        lines = target_file.readlines()
    
    with open(target_file_path, 'w') as target_file:
        for line in lines:
            if line.count(' ') <= threshold:
                target_file.write(line)

def main(data_name):
    base_path = "/scratch/jp6263/slackV2/" # Adjust base path as needed
    # Get files using data name
    train_data, test_data, _ = get_files_from_name(data_name, base_path)
    
    if not train_data or not test_data:
        print("Could not find both test and train data files. Please check the data name and path.")
        return
    else:
        print(f"Found train data: {train_data}")
        print(f"Found test data: {test_data}")
    
    # Step 1: Count spaces per line in the test data (source file)
    space_counts = count_spaces_per_line(test_data)
    
    # Step 2: Calculate the 95th percentile of space counts
    percentile_95 = np.percentile(space_counts, 95)
    
    # Step 3: Filter the train data (target file) by the calculated threshold
    filter_file_by_space_threshold(test_data, train_data, percentile_95)
    
    print(f"Filtered {train_data} with a space threshold of {percentile_95:.2f} (95th percentile of spaces in {test_data}).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_name>")
        sys.exit(1)
    data_name = sys.argv[1]
    main(data_name)
