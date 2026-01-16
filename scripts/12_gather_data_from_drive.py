import sys
import json
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.drive_downloads import download_drive_files

IDS_PATH = PROJECT_ROOT / "config" / "drive_ids.json"
DATA_ROOT = PROJECT_ROOT / "data"

SLEEP_SEC = 1.5
EXT = ".parquet"

def get_dest_folder(data_type, stage):
    """
    Returns the path: data/{stage}/{subfolder}
    data_type: 'text' or 'numerical'
    stage: 'raw', 'preprocessed', 'processed'
    """
    if data_type == "predictions":
        return DATA_ROOT / "processed" / "predictions"

    if data_type == "text_data":
        subfolder = "fnspid"
    else:
        subfolder = "numerical_data"

    if stage == "processed_and_linked":
        folder_stage = "processed"
    else:
        folder_stage = stage

    return DATA_ROOT / folder_stage / subfolder

def main():
    if not IDS_PATH.exists():
        print(f"Error: Config file not found at {IDS_PATH}")
        return
        
    with open(IDS_PATH, "r") as f:
        drive_ids = json.load(f)

    print("\nSelect Data Type:")
    print("1. Text Data (fnspid)")
    print("2. Numerical Data")
    print("3. Predictions")  
    
    type_choice = input(">>> ").strip().lower()
    
    if type_choice in ["1", "text", "text data"]:
        selected_type = "text_data"
    elif type_choice in ["2", "numerical", "numerical data"]:
        selected_type = "numerical_data"
    elif type_choice in ["3", "predictions", "prediction", "pred"]:  
        selected_type = "predictions"
    else:
        print("Invalid choice.")
        return


    if selected_type == "predictions":
        file_map = drive_ids[selected_type]
        dest_folder = get_dest_folder(selected_type, stage="processed")  

        print(f"\n--- Configuration ---")
        print(f"Type:   {selected_type}")
        print(f"Stage:  (none)")
        print(f"Target: {dest_folder}")

        prefix = ""

        print("\nStarting Download...")
        download_drive_files(
            file_map,
            dest_folder,
            prefix=prefix,
            sleep_sec=SLEEP_SEC,
            ext=EXT
        )

        print("\nDone.")
        return

    available_stages = list(drive_ids[selected_type].keys())
    
    if not available_stages:
        print(f"No datasets configured for {selected_type} yet.")
        return

    print(f"\nSelect Stage for {selected_type}:")
    for i, stage in enumerate(available_stages):
        print(f"{i+1}. {stage}")
        
    stage_choice = input(">>> ").strip()
    
    try:
        if stage_choice.isdigit():
            idx = int(stage_choice) - 1
            if 0 <= idx < len(available_stages):
                selected_stage = available_stages[idx]
            else:
                raise ValueError
        else:
            if stage_choice in available_stages:
                selected_stage = stage_choice
            else:
                raise ValueError
    except ValueError:
        print("Invalid stage selection.")
        return

    file_map = drive_ids[selected_type][selected_stage]
    dest_folder = get_dest_folder(selected_type, selected_stage)
    
    print(f"\n--- Configuration ---")
    print(f"Type:   {selected_type}")
    print(f"Stage:  {selected_stage}")
    print(f"Target: {dest_folder}")
    
    prefix = ""
    
    if selected_type == "text_data":
        if selected_stage == "processed_and_linked":
            prefix = "" 
        elif selected_stage in ["preprocessed", "processed"]:
            prefix = selected_stage

    elif selected_type == "numerical_data":
        prefix = "" 

    print("\nStarting Download...")
    download_drive_files(
        file_map,
        dest_folder,
        prefix=prefix,
        sleep_sec=SLEEP_SEC,
        ext=EXT
    )

    print("\nDone.")

if __name__ == "__main__":
    main()