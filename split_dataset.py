#!/usr/bin/env python3
"""
Dataset Splitter Script
Splits a dataset into train (80%) and validation (20%) sets.
Maintains class folder structure for PyTorch ImageFolder compatibility.

Usage:
    python split_dataset.py <source_dataset_path> <output_dataset_path>

Example:
    python split_dataset.py isd_dataset/ capstone_dataset/
    
This will create:
    capstone_dataset/
        train/
            class1/
            class2/
            ...
        val/
            class1/
            class2/
            ...
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    return [f for f in directory.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions]

def split_files(files: List[Path], train_ratio: float = 0.8) -> Tuple[List[Path], List[Path]]:
    """Split files into train and validation sets."""
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]

def create_directory_structure(output_path: Path, class_names: List[str]):
    """Create train/val directory structure."""
    for split in ['train', 'val']:
        for class_name in class_names:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

def copy_files(files: List[Path], destination_dir: Path):
    """Copy files to destination directory."""
    for file_path in files:
        shutil.copy2(file_path, destination_dir / file_path.name)

def split_dataset(source_path: Path, output_path: Path, train_ratio: float = 0.8, seed: int = 42):
    """Split dataset into train and validation sets."""
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")
    
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_path}")
    
    class_names = [d.name for d in class_dirs]
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    create_directory_structure(output_path, class_names)
    
    total_train = 0
    total_val = 0
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all image files for this class
        image_files = get_image_files(class_dir)
        
        if not image_files:
            print(f"  Warning: No image files found in {class_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        
        # Split into train and validation
        train_files, val_files = split_files(image_files, train_ratio)
        
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Copy files to respective directories
        train_dir = output_path / 'train' / class_name
        val_dir = output_path / 'val' / class_name
        
        copy_files(train_files, train_dir)
        copy_files(val_files, val_dir)
        
        total_train += len(train_files)
        total_val += len(val_files)
    
    print(f"\n✅ Dataset split completed!")
    print(f"📊 Summary:")
    print(f"  Total images: {total_train + total_val}")
    print(f"  Training set: {total_train} images ({total_train/(total_train+total_val)*100:.1f}%)")
    print(f"  Validation set: {total_val} images ({total_val/(total_train+total_val)*100:.1f}%)")
    print(f"  Classes: {len(class_names)}")
    print(f"\n📁 Output structure:")
    print(f"  {output_path}/")
    print(f"    train/")
    for class_name in class_names:
        train_count = len(list((output_path / 'train' / class_name).glob('*')))
        print(f"      {class_name}/ ({train_count} images)")
    print(f"    val/")
    for class_name in class_names:
        val_count = len(list((output_path / 'val' / class_name).glob('*')))
        print(f"      {class_name}/ ({val_count} images)")

def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/validation sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python split_dataset.py isd_dataset/ capstone_dataset/
    python split_dataset.py my_data/ split_data/ --train-ratio 0.7 --seed 123
        """
    )
    
    parser.add_argument('source', type=str, 
                       help='Source dataset directory containing class folders')
    parser.add_argument('output', type=str, 
                       help='Output directory for train/val split')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducible splits (default: 42)')
    
    args = parser.parse_args()
    
    if not (0 < args.train_ratio < 1):
        raise ValueError("train-ratio must be between 0 and 1")
    
    source_path = Path(args.source)
    output_path = Path(args.output)
    
    try:
        split_dataset(source_path, output_path, args.train_ratio, args.seed)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())