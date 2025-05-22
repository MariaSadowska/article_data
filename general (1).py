import pandas as pd
import os
import argparse
import threading
import subprocess
import ast  # To safely evaluate list from string

# =============================================================================
# Set working directory
# =============================================================================
base_working_directory = '/lu/topola/home/mariasad/MS'
os.chdir(base_working_directory)
data_file_path = os.path.join(base_working_directory, 'data_noise.xlsx')

# =============================================================================
# Argument parser setup
# =============================================================================
parser = argparse.ArgumentParser(description='Run time-series classification scripts based on a list of numbers.')
parser.add_argument('numbers', type=str, help='The list of g_numbers to process (e.g., "[1,2,4]")')
args = parser.parse_args()

# =============================================================================
# Convert the string argument to a list of integers or range
# =============================================================================
def parse_numbers(input_str):
    if ':' in input_str:  # Check if the input contains a range
        # Remove square brackets and split by ':'
        start, end = map(int, input_str.strip('[]').split(':'))
        return list(range(start, end + 1))  # Generate the list using range
    else:
        # Convert the string argument to a list of integers (e.g., "[1,2,4]")
        return ast.literal_eval(input_str)

g_numbers = parse_numbers(args.numbers)

# =============================================================================
# Load Excel file
# =============================================================================
data = pd.read_excel(data_file_path)

# =============================================================================
# Function to run the test script with a specific g_number
# =============================================================================
def run_script(g_number, task_id, num_tasks):
    script_name = "test1_noise_param_grid.py"
    subprocess.run(["python3.9", script_name, str(g_number)])
    
    
# =============================================================================
# Iterate over g_numbers and run the test script
# =============================================================================
threads = []
for g_number in g_numbers:
    if g_number in data['g_number'].values:
        print(f"Starting thread for g_number {g_number}")
        thread = threading.Thread(target=run_script, args=(g_number, 0, 0))
        threads.append(thread)
        thread.start()
    else:
        print(f"g_number {g_number} not found in data.xlsx")

for thread in threads:
    thread.join()

print("All threads have finished execution.")