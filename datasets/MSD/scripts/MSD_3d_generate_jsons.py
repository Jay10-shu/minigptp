# File: /home/sj/MiniGPT-Pancreas/datasets/scripts/generate_report_data.py
import os
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm

def get_3d_bbox_from_mask(mask_array):
    """
    Calculates the 3D bounding box from a 3D numpy array (mask).
    The bounding box is returned as [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    # Find the coordinates of all non-zero voxels (the mask)
    indices = np.where(mask_array > 0)
    
    # If the mask is empty, return None
    if len(indices[0]) == 0:
        return None

    # Determine the min and max for each axis
    x_min, x_max = int(np.min(indices[0])), int(np.max(indices[0]))
    y_min, y_max = int(np.min(indices[1])), int(np.max(indices[1]))
    z_min, z_max = int(np.min(indices[2])), int(np.max(indices[2]))
    
    return [x_min, y_min, z_min, x_max, y_max, z_max]

def process_patient_data(root_dir, output_filename="pancreas_data_all.json"):
    """
    Processes patient data from a specified directory structure, extracts metadata,
    and saves it to a JSON file.

    Args:
        root_dir (str): The root directory containing patient folders.
                        e.g., '/home/sj/MiniGPT-Pancreas/datasets/Report/toydata'
        output_filename (str): The name for the output JSON file.
    """
    all_patient_data = []
    
    # Get all patient directories within the root directory
    patient_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for patient_id in tqdm(patient_folders, desc='Processing Patient Data'):
        patient_path = os.path.join(root_dir, patient_id)
        
        # Initialize paths for the required files
        raw_image_path = None
        pancreas_mask_path = None
        tumor_mask_path = None

        # Find the specific files within each patient's folder
        for filename in os.listdir(patient_path):
            if filename.endswith((".nii", ".nii.gz")):
                if "_p." in filename:
                    pancreas_mask_path = os.path.join(patient_path, filename)
                elif "_t." in filename:
                    tumor_mask_path = os.path.join(patient_path, filename)
                elif "_CT_" in filename:  # Assumes original CT contains "_CT_"
                    raw_image_path = os.path.join(patient_path, filename)

        # Ensure essential files are found before proceeding
        if not all([raw_image_path, pancreas_mask_path]):
            print(f"Warning: Skipping patient {patient_id} due to missing raw image or pancreas mask.")
            continue

        try:
            # 1. Extract label from the folder name (first 4 characters)
            label = patient_id[:4]

            # 2. Calculate the pancreas bounding box
            p_mask_img = nib.load(pancreas_mask_path)
            pancreas_bbox = get_3d_bbox_from_mask(p_mask_img.get_fdata())
            if pancreas_bbox is None:
                print(f"Warning: Skipping patient {patient_id} because the pancreas mask is empty.")
                continue

            # 3. Calculate the tumor bounding box (if a tumor mask exists)
            tumor_bbox = [0, 0, 0, 0, 0, 0]  # Default value if no tumor
            if tumor_mask_path:
                t_mask_img = nib.load(tumor_mask_path)
                found_tumor_bbox = get_3d_bbox_from_mask(t_mask_img.get_fdata())
                if found_tumor_bbox:
                    tumor_bbox = found_tumor_bbox
            
            # 4. Get the shape of the original CT image
            raw_image_nifti = nib.load(raw_image_path)
            shape = raw_image_nifti.get_fdata().shape

            # 5. Assemble all information into a dictionary
            patient_info = {
                "patient_id": patient_id,
                "label": label,
                "volume_name": os.path.basename(raw_image_path),
                "image_path": raw_image_path,
                "shape": [int(s) for s in shape],
                "pancreas_bbox_3d": pancreas_bbox,
                "tumor_bbox_3d": tumor_bbox
            }
            all_patient_data.append(patient_info)

        except Exception as e:
            print(f"An error occurred while processing patient {patient_id}: {e}")
            continue
            
    # 6. Save the aggregated data to a JSON file
    # The output will be in a new 'annotations' directory at the same level as 'toydata'
    output_dir = os.path.join(os.path.dirname(root_dir), "annotations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(all_patient_data, f, indent=4)
        
    print(f"\nProcessing complete. Processed {len(all_patient_data)} patients.")
    print(f"Data saved to: {output_path}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # Set the path to your main data directory here
    data_root_directory = '/home/sj/MiniGPT-Pancreas/datasets/Report/toydata'
    process_patient_data(data_root_directory)