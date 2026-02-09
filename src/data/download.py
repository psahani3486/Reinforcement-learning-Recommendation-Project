"""
Download MovieLens dataset.

This module handles downloading and extracting the MovieLens 1M dataset.
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


# Dataset URLs
MOVIELENS_URLS = {
    'movielens-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'movielens-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'movielens-20m': 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
}


def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Local path to save the file
        chunk_size: Size of chunks for streaming download
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")


def download_movielens(dataset: str = 'movielens-1m', data_dir: str = 'data/raw') -> str:
    """
    Download and extract MovieLens dataset.
    
    Args:
        dataset: Which MovieLens version ('movielens-1m', 'movielens-100k', 'movielens-20m')
        data_dir: Directory to save the data
        
    Returns:
        Path to the extracted dataset directory
    """
    if dataset not in MOVIELENS_URLS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(MOVIELENS_URLS.keys())}")
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset specific paths
    url = MOVIELENS_URLS[dataset]
    zip_filename = url.split('/')[-1]
    zip_path = data_path / zip_filename
    
    # Extract folder name (ml-1m, ml-100k, etc.)
    extract_name = zip_filename.replace('.zip', '')
    extract_path = data_path / extract_name
    
    # Check if already downloaded
    if extract_path.exists():
        print(f"Dataset already exists at {extract_path}")
        return str(extract_path)
    
    # Download
    print(f"Downloading {dataset} from {url}...")
    download_file(url, str(zip_path))
    
    # Extract
    extract_zip(str(zip_path), str(data_path))
    
    # Remove zip file to save space
    os.remove(zip_path)
    print(f"Removed zip file. Dataset saved to {extract_path}")
    
    return str(extract_path)


def verify_dataset(data_path: str, dataset: str = 'movielens-1m') -> bool:
    """
    Verify that the dataset was downloaded correctly.
    
    Args:
        data_path: Path to the dataset directory
        dataset: Which dataset version
        
    Returns:
        True if verification passes
    """
    data_dir = Path(data_path)
    
    if dataset == 'movielens-1m':
        required_files = ['ratings.dat', 'movies.dat', 'users.dat']
    elif dataset == 'movielens-100k':
        required_files = ['u.data', 'u.item', 'u.user']
    else:
        required_files = ['ratings.csv', 'movies.csv']
    
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"Missing required file: {file}")
            return False
    
    print("Dataset verification passed!")
    return True


if __name__ == '__main__':
    # Download MovieLens 1M by default
    import argparse
    
    parser = argparse.ArgumentParser(description='Download MovieLens dataset')
    parser.add_argument('--dataset', type=str, default='movielens-1m',
                        choices=['movielens-1m', 'movielens-100k', 'movielens-20m'],
                        help='Which MovieLens dataset to download')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory to save the data')
    
    args = parser.parse_args()
    
    data_path = download_movielens(args.dataset, args.data_dir)
    verify_dataset(data_path, args.dataset)
