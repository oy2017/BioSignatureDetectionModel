import os
import subprocess
import shutil
import sys

fill_gas = "H2"
num_sets = 5

print(f"--- Generating {num_sets} Independent Test Sets for {fill_gas} ---")

for i in range(1, num_sets + 1):
    print(f"\n[Generation {i}/{num_sets}]")

    # Run the generation script
    # We rely on it producing 'multirex_spectra_H2_test.parquet'
    subprocess.run([sys.executable, "generate_multiverse_data.py", fill_gas, "--purpose", "test"], check=True)
    
    # Rename the output file
    default_name = f"multirex_spectra_{fill_gas}_test.parquet"
    new_name = f"multirex_spectra_{fill_gas}_test_set_{i}.parquet"
    
    if os.path.exists(default_name):
        if os.path.exists(new_name):
            os.remove(new_name)
        shutil.copy(default_name, new_name) # Copy instead of rename
        print(f"Saved copy of test set to: {new_name}")
    else:
        print(f"Error: Expected output file {default_name} not found!")
        exit(1)

print("\n--- All 5 Test Sets Generated Successfully ---")

