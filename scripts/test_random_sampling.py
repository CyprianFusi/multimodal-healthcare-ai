"""
Test that ROCO random sampling returns different images each time

Usage:
    python scripts/test_random_sampling.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal.processors.roco_loader import ROCODataset


def main():
    print("="*70)
    print("ROCO Random Sampling Test")
    print("="*70)
    print("\nThis test verifies that get_random_sample() returns different")
    print("images each time it's called.\n")

    # Load dataset
    roco = ROCODataset('./data/roco')

    # Get 10 random samples
    print("Getting 10 random samples from validation set:")
    print("-"*70)

    roco_ids = []
    for i in range(10):
        sample = roco.get_random_sample(split='val')
        roco_id = sample['metadata']['roco_id']
        image_id = sample['metadata']['image_id'][:50]
        roco_ids.append(roco_id)

        print(f"{i+1:2}. {roco_id:15} | {image_id}")

    # Check for duplicates
    unique_ids = len(set(roco_ids))
    total_ids = len(roco_ids)

    print("\n" + "="*70)
    if unique_ids == total_ids:
        print(f"SUCCESS: All {total_ids} samples were unique!")
        print("Random sampling is working correctly.")
    else:
        duplicates = total_ids - unique_ids
        print(f"WARNING: Found {duplicates} duplicate(s) out of {total_ids} samples")
        print(f"Unique samples: {unique_ids}")
        print("\nNote: Some duplicates might occur by random chance with small samples.")
        print("If ALL samples are the same, there's a problem.")

    print("="*70)

    return 0 if unique_ids > 1 else 1


if __name__ == '__main__':
    sys.exit(main())
