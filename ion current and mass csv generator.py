"""
This script calculates ion current for a Langmuir probe and integrates mass measurements from a Quartz Crystal Microbalance (QCM).
It processes CSV files from an oscilloscope and an Excel file containing experimental parameters.

Purpose:
    - Calculate ion current from voltage measurements.
    - Integrate mass measurements from QCM.
    - Output aggregated results for analysis.

Assumptions:
    - CSV files are named as `data_XXX.csv`, where `XXX` is a numeric index.
    - The Excel file has a specific structure (see column documentation).

Usage:
    1. Change the folder_path variable to where you placed your CSV files.
    2. Customize the columns for the excel_config
    3. Configure the script 
    (e.g., adjust `voltage_to_ioncurrent` for different resistors or change the time to average over t0 and t1).


Limitations:
    - Assumes a specific CSV format for oscilloscope data.
    - Mass calculation is only valid for QCM measurements.

"""
#%%

import os
import pandas as pd
import numpy as np

# --- Configuration ---

folder_path = r"/media/sf_Junk.Paul/Measurements/qcm+probe/20cm_QCM+probe_N2_singleshots"

#lab protocol page, suggested format to not have to change anything of the code at the bottom
measurements_path = r"/media/sf_Junk.Paul/Measurements/measurements.xlsx"  
out_per_group = "/home/bruh/Documents/20_cm_ion_current.csv"

# CSV processing parameters in folder_path
# Averaging the ion_current measured with a oscilloscope from 0ms to 1ms 

skiprows = 11 # to remove unnecessary header (change as needed)
time_col = 0  # Column index for time in CSV (should always be first column)
data_col = 3  # Column index for data in CSV (Data column to average over)
t0 = 0.0      
t1 = 1e-3    


# Conversion factor of probe circuit
voltage_to_ioncurrent = 1/400



# Excel column indices (0-based, A->0; B->1 ...), Change as needed. Here also adding Mass measured with QCM
excel_config = {
    "pressure_col": 5,      # (Work_Pa)
    "magfield_col": 9,      # (Mag_acc) Here I used an additional solenoid and i want to keep that information
    "mass_col": 17,         # (Mass (g))
    "suffix_col": 3,        # Column index for file suffix in folder
}

# Excel row range (0-based index)
excel_row_range = {
    "start_row": 50,
    "end_row": 91,   
}


#%%
# --- Functions ---
def build_from_range(df, config, start_row, end_row):
    """
    Builds a group_spec string from a specific row range in the Excel DataFrame.

    Parameters:
        df (pd.DataFrame): The Excel data as a DataFrame.
        config (dict): A dictionary with keys for column indices (e.g., "pressure_col", "magfield_col").
        start_row (int): The starting row index (0-based).
        end_row (int): The ending row index (0-based).

    Returns:
        str: A semicolon-separated string of group labels and suffixes (e.g., "1.0_2.0_3.0e-8:1,2,3").

    Raises:
        ValueError: If the DataFrame is empty or the row range is invalid.
    """
    group_spec = []
    for _, row in df.iloc[start_row:end_row+1].iterrows():
        pressure = row.iloc[config["pressure_col"]] if config["pressure_col"] is not None else "?"
        magfield = row.iloc[config["magfield_col"]] if config["magfield_col"] is not None else "?"
        mass = row.iloc[config["mass_col"]] if config["mass_col"] is not None else "?"
        suffix = row.iloc[config["suffix_col"]]

        # Skip if suffix is NaN or invalid
        if pd.isna(suffix):
            continue

        label = f"{pressure}_{magfield}_{mass}"
        group_spec.append(f"{label}:{suffix}")
    return ";".join(group_spec)


def average_file(filepath, skiprows, time_col, data_col, t0, t1):
    """
    Reads a CSV file, skips rows, and averages data between t0 and t1.
    Returns both the mean and the standard deviation of the data within the specified time range.
    """
    df = pd.read_csv(filepath, skiprows=skiprows, header=None)
    mask = (df.iloc[:, time_col] >= t0) & (df.iloc[:, time_col] <= t1)
    data_within_range = df.loc[mask, df.columns[data_col]]
    avg = data_within_range.mean()
    std = data_within_range.std()
    return avg, std

def extract_index(filename):
    """
    Extracts the numeric index from the filename.
    Assumes filenames are like 'data_001.csv'.
    """
    basename = os.path.basename(filename)
    index = basename.split("_")[1].split(".")[0]
    return int(index)

def parse_group_spec(group_spec):
    """
    Parses the group_spec string into a dictionary of groups.
    Skips invalid or missing suffixes.
    """
    groups = {}
    for item in group_spec.split(";"):
        if ":" not in item:
            continue  # Skip malformed entries
        label, suffixes_str = item.split(":")
        suffixes = []
        for suffix in suffixes_str.split(","):
            try:
                suffixes.append(int(suffix))
            except ValueError:
                continue  # Skip invalid suffixes
        if suffixes:  # Only add if there are valid suffixes
            groups[label] = suffixes
    return groups


# --- Main Logic ---
def main_function():

    # Read Excel file and build group_spec
    df_excel = pd.read_excel(measurements_path, header=None)
    group_spec = build_from_range(
        df_excel, excel_config, excel_row_range["start_row"], excel_row_range["end_row"]
    )
    groups = parse_group_spec(group_spec)

    # Process CSV files
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            index = extract_index(filename)
            avg, std = average_file(filepath, skiprows, time_col, data_col, t0, t1)
            # Find the group for this index
            for label, indices in groups.items():
                if index in indices:
                    results.append({
                        "group": label,
                        "mean": avg,
                        "std": std,
                        "index": index,
                    })
                    break

    # Aggregate results by group
    aggregated = []
    for label, indices in groups.items():
        group_data = [r for r in results if r["group"] == label]
        if group_data:
            means = [r["mean"] for r in group_data]
            stds = [r["std"] for r in group_data]
            mean = np.mean(means)
            std = np.mean(stds)  # This is the mean of the standard deviations
            count = len(group_data)
            ioncurrent = mean * voltage_to_ioncurrent
            ioncurrent_std = std * voltage_to_ioncurrent
            # Split the group label into pressure, magfield, and mass
            pressure, magfield, mass = label.split("_")
            # Apply the limiter for pressure
            pressure = float(pressure) if pressure != "?" else 0
            pressure = pressure if pressure >= 0.001 else 0
            aggregated.append({
                "pressure": pressure,
                "magfield": magfield,
                "mass": mass,
                "mean": mean,
                "std": std,
                "count": count,
                "ioncurrent": ioncurrent,
                "ioncurrent_std": ioncurrent_std,
            })

    # Sort by pressure and magnetic field
    aggregated.sort(key=lambda x: (
        float(x["pressure"]) if x["pressure"] != "?" else 0,
        float(x["magfield"]) if x["magfield"] != "?" else 0,
    ))

    # Save to CSV
    df_out = pd.DataFrame(aggregated)
    df_out = df_out.rename(columns={
        "pressure": "Pressure (Pa)",
        "magfield": "MagField (T)",
        "mass": "Mass (g)",
        "mean": "Mean",
        "std": "Std",
        "count": "Count",
        "ioncurrent": "IonCurrent (A)",
        "ioncurrent_std": "IonCurrent_Std (A)",
    })
    df_out.to_csv(out_per_group, index=False)
    print(f"Results saved to {out_per_group}")

if __name__ == "__main__":
    main_function()
    
    
    




