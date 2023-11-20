import numpy as np
import os

# Define the path to the folder containing .npy files
processed_data_folder = 'C:\\Users\\erick\\Desktop\\GT_Study\\ECE_6122\\Final_Project\\Audio_data\\cats_dogs\\Processed'

# Set the number of files to load
num_files_to_load = 10

# Retrieve all .npy files in the folder
npy_files = [f for f in os.listdir(processed_data_folder) if f.endswith('.npy')]

# Define the path for the report file
report_file_path = 'C:\\Users\\erick\\Desktop\\GT_Study\\ECE_6122\\Final_Project\\Audio_data\\cats_dogs\\Processed\\Report\\npy_report.txt'

# Open the report file for writing
with open(report_file_path, 'w') as report_file:
    # Load and write the contents of the first few files
    for file in npy_files[:num_files_to_load]:
        file_path = os.path.join(processed_data_folder, file)
        # Set allow_pickle=True to load object arrays
        data = np.load(file_path, allow_pickle=True)
        report = f"Contents of {file}:\n{data}\n\n"
        # Write the report to the file instead of printing it
        report_file.write(report)
