# -*- coding: utf-8 -*-
"""
Created on Fri May  2 18:51:25 2025

@author: marle
"""

import os
import pandas as pd

# Folder path with CSV-files
input_folder = ""
output_filename = ""
output_path = os.path.join(input_folder, output_filename)

# List for TPs entries
tps_entries = []

for filename in os.listdir(input_folder):
    if filename.endswith("_original.csv"):
        filepath = os.path.join(input_folder, filename)

        # read csv file
        df = pd.read_csv(filepath, sep=",", engine="python", skipinitialspace=True)

        # Check for required columns
        if "SampleID" in df.columns and "ObjectID" in df.columns and "x" in df.columns and "y" in df.columns:
            sample_id = df.loc[0, "SampleID"]
            object_id = df.loc[0, "ObjectID"]
            full_id = f"{sample_id}_{object_id}"

            # Extract coordinate columns only
            coords = df[["x", "y"]].copy()

            # The leftmost and rightmost points along the X-axis
            leftmost_idx = coords["x"].idxmin()
            rightmost_idx = coords["x"].idxmax()

            leftmost = coords.loc[leftmost_idx]
            rightmost = coords.loc[rightmost_idx]

            # New order: left point, right point, then all original points
            extended_coords = pd.concat([
                pd.DataFrame([leftmost]),  # LM1
                pd.DataFrame([rightmost]), # LM2
                coords                     # Original Landmarks
            ], ignore_index=True)

            # Make sure there are exactly 102 points 
            if len(extended_coords) != 102:
                print(f"❌ Error: File '{filename}' does not contain exactly 102 points after the expansion.")
                continue

            # Create a TPS block
            tps_block = ["LM=102"]
            tps_block += [f"{row['x']} {row['y']}" for _, row in extended_coords.iterrows()]
            tps_block.append(f"ID={full_id}")
            tps_block.append("SCALE=0.69")

            # Add to total
            tps_entries.append("\n".join(tps_block))
        else:
            print(f"⚠️ file '{filename}' does not include all the necessary columns (SampleID, ObjectID, X, Y).")

# Write to TPS file
if tps_entries:
    with open(output_path, "w") as f:
        f.write("\n".join(tps_entries))  # Two lines of space between entries
    print(f"✅ TPS file saved as: {output_path}")
else:
    print("❌ No valid CSV files were found for TPS generation.")
