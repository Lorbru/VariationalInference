# import 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from scipy.interpolate import interp1d
import ast
import matplotlib.pyplot as plt
import json 

def interpolate_data(df, n_trajectories,n_points=50):
    """
    Interpolate trajectories by resampling a subset of trajectories and creating evenly spaced points.

    Args:
    - df (pandas DataFrame): DataFrame containing trajectory data, with a column 'POLYLINE' containing 
      the list of (longitude, latitude) points for each trajectory.
    - n_trajectories (int): Number of trajectories to sample from the DataFrame for interpolation.
    - n_points (int, optional): Number of points to generate for each interpolated trajectory. Default is 50.

    Returns:
    - interpolated (numpy array): A 2D array of shape [n_samples, n_points * 2], where each sample is 
      an interpolated trajectory with `n_points` coordinates (longitude, latitude).
    - filtered_data (pandas DataFrame): DataFrame containing only the valid sampled trajectories from the 
      original dataset (after filtering out invalid trajectories with fewer than 2 points).
    """
    # Sample n_trajectories trajectories
    np.random.seed(42)
    sampled_indices = np.random.choice(df.index, n_trajectories, replace=False)
    data = df.iloc[sampled_indices].copy() 

    # Convert string representation of polyline to actual lists
    data['POLYLINE'] = data['POLYLINE'].apply(ast.literal_eval)
    
    # Initialize array for interpolated trajectories
    interpolated = np.full((len(data), n_points, 2), np.nan)
    
    valid_count = 0
    valid_indices = []  # Store valid trajectory indices

    for idx, trajectory in enumerate(data['POLYLINE']):
        if len(trajectory) < 2:
            continue  # Skip invalid trajectories

        # Convert to numpy array
        traj_array = np.array(trajectory)

        # Create interpolation functions for longitude and latitude
        path_length = np.linspace(0, 1, len(traj_array))

        lon_interp = interp1d(path_length, traj_array[:, 0], kind='linear')
        lat_interp = interp1d(path_length, traj_array[:, 1], kind='linear')
        
        # Generate new evenly spaced points
        new_path = np.linspace(0, 1, n_points)
        
        # Interpolate coordinates
        interpolated[valid_count, :, 0] = lon_interp(new_path) # Longitude
        interpolated[valid_count, :, 1] = lat_interp(new_path) # Latitude
        valid_count += 1
        valid_indices.append(data.index[idx])
    
    # Remove empty entries (both zeros and NaNs)
    interpolated = interpolated[:valid_count]  # Trim to actual valid trajectories
    filtered_data = data.loc[valid_indices]
    
    # Flatten to (n_samples, n_points*2) shape
    flattened = interpolated.reshape(len(interpolated), -1)

    return flattened, filtered_data

def visualize_interpolation(original, interpolated, n_examples=3):
    """
    Visualize a comparison between original and interpolated trajectories.

    Args:
    - original (pandas Series or DataFrame): The original trajectory data, where each entry contains a 
      list of (longitude, latitude) points representing a trajectory.
    - interpolated (numpy array): A 2D numpy array of shape [n_samples, n_points, 2] containing the 
      interpolated trajectories.
    - n_examples (int, optional): The number of example trajectories to visualize. Default is 3.

    Returns:
    - None: This function displays a plot with the original and interpolated trajectories.
    """
    plt.figure(figsize=(15, 5))
    for i in range(n_examples):
        # Original trajectory
        plt.subplot(2, n_examples, i+1)
        orig = np.array(original.iloc[i])
        plt.plot(orig[:, 0], orig[:, 1], 'b-o', markersize=3)
        plt.title(f"Original Trajectory {i+1}")
        
        # Interpolated trajectory
        plt.subplot(2, n_examples, i+1+n_examples)
        intrp = interpolated[i].reshape(-1, 2)
        plt.plot(intrp[:, 0], intrp[:, 1], 'r-o', markersize=3)
        plt.title(f"Interpolated {i+1}")
    
    plt.tight_layout()
    plt.show()

def save_interpolated_to_csv(interpolated, save_path):
    """
    Save interpolated trajectories into a CSV file.

    Args:
    - interpolated (numpy array): Shape [n_samples, n_points, 2], interpolated trajectory data.
    - save_path (str): Path where to save the CSV file.

    Returns:
    - None: The function saves the interpolated trajectories to the specified CSV file and prints a confirmation.
    """
    # Build the dataframe
    records = []
    
    for i in range(len(interpolated)):
        polyline = interpolated[i].tolist()  # Convert numpy array to list of points
        
        # Serialize polyline as JSON string
        polyline_str = json.dumps(polyline)
        
        records.append({
            'POLYLINE': polyline_str
        })
    
    # Create a dataframe
    result_df = pd.DataFrame(records)
    
    # Save to CSV
    result_df.to_csv(save_path, index=False)
    print(f"Interpolated data saved to {save_path}")

class LoadDataset(Dataset):

    def __init__(self):
        return
    
    