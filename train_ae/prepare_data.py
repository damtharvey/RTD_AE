#!/usr/bin/env python3
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import tarfile
from tqdm import tqdm
import cv2
import glob
import shutil
import urllib.request

def download_coil20():
    """Download and extract COIL-20 dataset."""
    # Try different URLs for the dataset
    urls = [
        "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.tar",
        "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.tar",
        "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/coil-20-proc.tar"
    ]
    
    data_dir = "data/COIL-20"
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "coil-20-proc.tar")
    
    # Try each URL until one works
    for url in urls:
        try:
            print(f"Trying to download from {url}")
            # Download the dataset
            print("Downloading COIL-20 dataset...")
            urllib.request.urlretrieve(url, tar_path)
            print("Download successful")
            break
        except Exception as e:
            print(f"Failed to download from {url}: {str(e)}")
            continue
    else:
        raise ValueError("Failed to download the dataset from all URLs")
    
    # Verify the tar file exists and is not empty
    if not os.path.exists(tar_path):
        raise ValueError("Tar file was not downloaded")
    if os.path.getsize(tar_path) == 0:
        raise ValueError("Downloaded tar file is empty")
    
    # Extract the dataset
    print("Extracting dataset...")
    extract_dir = os.path.join(data_dir, "coil-20-proc")
    
    # Remove existing directory if it exists
    if os.path.exists(extract_dir):
        print(f"Removing existing directory: {extract_dir}")
        shutil.rmtree(extract_dir)
    
    os.makedirs(extract_dir, exist_ok=True)
    
    # List contents of tar file
    try:
        with tarfile.open(tar_path, 'r') as tar:
            print("Contents of tar file:")
            for member in tar.getmembers():
                print(f"  {member.name}")
            
            # Extract all files
            print("\nExtracting files...")
            tar.extractall(path=extract_dir)
    except Exception as e:
        print(f"Error extracting tar file: {str(e)}")
        raise
    
    # Verify extraction
    print("\nVerifying extracted files...")
    extracted_files = glob.glob(os.path.join(extract_dir, "**", "*.png"), recursive=True)
    print(f"Found {len(extracted_files)} PNG files in {extract_dir}")
    
    if not extracted_files:
        print("No PNG files found after extraction. Listing directory contents:")
        for root, dirs, files in os.walk(extract_dir):
            print(f"\nDirectory: {root}")
            for file in files:
                print(f"  {file}")
    else:
        print("\nFirst few files found:")
        for f in extracted_files[:5]:
            print(f"  {f}")
    
    return extract_dir

def load_and_preprocess_data(data_path):
    """Load and preprocess COIL-20 images."""
    print("Loading and preprocessing images...")
    images = []
    labels = []
    
    # Get all PNG files in the directory and subdirectories
    image_files = glob.glob(os.path.join(data_path, "**", "*.png"), recursive=True)
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print(f"Directory contents of {data_path}:")
        for root, dirs, files in os.walk(data_path):
            print(f"\nDirectory: {root}")
            for file in files:
                print(f"  {file}")
        raise ValueError(f"No PNG files found in {data_path}")
    
    # Load all images
    for img_path in tqdm(image_files):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # Resize image
            img = cv2.resize(img, (128, 128))
            
            # Flatten and normalize
            img_flat = img.flatten().astype(np.float32)
            img_flat = img_flat / 255.0  # Normalize to [0, 1]
            
            # Extract object ID from filename
            filename = os.path.basename(img_path)
            obj_id = int(filename.split('__')[0][3:]) - 1  # Convert to 0-based index
            
            images.append(img_flat)
            labels.append(obj_id)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    if not images:
        raise ValueError("No images were successfully loaded")
    
    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels)
    
    print(f"Loaded {len(images)} images with shape {X.shape}")
    
    # Normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def load_prepared_data():
    """Load the prepared COIL-20 dataset."""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(current_dir, "data/COIL-20/prepared")
    
    print(f"Looking for data in: {data_dir}")
    
    # Check if prepared data exists
    if not os.path.exists(data_dir):
        print(f"Directory contents of {os.path.dirname(data_dir)}:")
        for root, dirs, files in os.walk(os.path.dirname(data_dir)):
            print(f"\nDirectory: {root}")
            for file in files:
                print(f"  {file}")
        raise ValueError(f"Prepared data directory {data_dir} does not exist")
    
    # Load the data
    train_data_path = os.path.join(data_dir, "train_data.npy")
    test_data_path = os.path.join(data_dir, "test_data.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")
    
    # Check which files exist
    missing_files = []
    for path in [train_data_path, test_data_path, train_labels_path, test_labels_path]:
        if not os.path.exists(path):
            missing_files.append(os.path.basename(path))
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        raise ValueError("Some prepared data files are missing")
    
    # Load the data
    print("Loading data files...")
    X_train = np.load(train_data_path)
    X_test = np.load(test_data_path)
    y_train = np.load(train_labels_path)
    y_test = np.load(test_labels_path)
    
    print(f"Loaded prepared data:")
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load prepared data
    X_train, X_test, y_train, y_test = load_prepared_data()
    
    print("Data loading completed successfully")

if __name__ == "__main__":
    main() 