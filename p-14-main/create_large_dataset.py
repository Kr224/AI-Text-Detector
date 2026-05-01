# create_large_dataset.py
"""
Script to create a large dataset for training
"""

import os
import sys
from data_collector import DataCollector


def create_dataset():
    print("=" * 60)
    print("LARGE DATASET CREATION")
    print("=" * 60)

    collector = DataCollector()

    print("\nOptions:")
    print("1. Create new large dataset (1000+ samples)")
    print("2. Augment existing dataset")
    print("3. Create very large dataset (5000+ samples)")

    choice = input("\nEnter choice (1-3): ")

    if choice == '1':
        size = int(input("Enter number of samples (e.g., 1200): ") or "1200")
        dataset = collector.create_large_dataset(size, 'data/processed/large_dataset.csv')
        print(f"\nCreated dataset with {len(dataset)} samples")

    elif choice == '2':
        input_file = input(
            "Input file path [data/processed/labeled_dataset.csv]: ") or "data/processed/labeled_dataset.csv"
        output_file = input(
            "Output file path [data/processed/augmented_dataset.csv]: ") or "data/processed/augmented_dataset.csv"
        dataset = collector.augment_existing_dataset(input_file, output_file)
        print(f"\nAugmented dataset saved to {output_file}")

    elif choice == '3':
        size = int(input("Enter number of samples (e.g., 5000): ") or "5000")
        dataset = collector.create_large_dataset(size, 'data/processed/very_large_dataset.csv')
        print(f"\nCreated very large dataset with {len(dataset)} samples")

    else:
        print("Invalid choice")

    # Show statistics
    if 'dataset' in locals():
        human_count = len(dataset[dataset['label'] == 0])
        ai_count = len(dataset[dataset['label'] == 1])
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(dataset)}")
        print(f"Human samples: {human_count} ({human_count / len(dataset) * 100:.1f}%)")
        print(f"AI samples: {ai_count} ({ai_count / len(dataset) * 100:.1f}%)")

        # Show sample texts
        print(f"\nSample Human Text:")
        print(dataset[dataset['label'] == 0].iloc[0]['text'][:200] + "...")
        print(f"\nSample AI Text:")
        print(dataset[dataset['label'] == 1].iloc[0]['text'][:200] + "...")


if __name__ == "__main__":
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/human', exist_ok=True)
    os.makedirs('data/ai', exist_ok=True)

    create_dataset()