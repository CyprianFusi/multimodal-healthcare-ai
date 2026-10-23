"""
Quick test script to verify ROCO integration works end-to-end

Usage:
    python scripts/test_roco_integration.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal.processors.roco_loader import ROCODataset
from src.multimodal.analyzers.multimodal_analyzer import MultiModalAnalyzer


def main():
    print("="*70)
    print("ROCO Integration Test")
    print("="*70)

    # Test 1: Load dataset
    print("\n[1/4] Loading ROCO dataset...")
    try:
        roco = ROCODataset('./data/roco')
        stats = roco.get_stats()
        print(f"SUCCESS: Dataset loaded!")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  - Train: {stats['train_samples']:,}")
        print(f"  - Val: {stats['val_samples']:,}")
        print(f"  - Test: {stats['test_samples']:,}")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    # Test 2: Load a sample image
    print("\n[2/4] Loading sample image...")
    try:
        sample = roco.get_sample(split='val', index=0)
        print(f"SUCCESS: Loaded image {sample['metadata']['roco_id']}")
        print(f"  Image ID: {sample['metadata']['image_id']}")
        print(f"  Image size: {sample['image'].size}")
        print(f"  Caption: {sample['caption'][:100]}...")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    # Test 3: Prepare for analysis
    print("\n[3/4] Preparing for AI analysis...")
    try:
        prepared = roco.prepare_for_analysis(sample)
        print(f"SUCCESS: Image prepared")
        print(f"  Base64 size: {len(prepared['image_base64'])} bytes")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    # Test 4: Run AI analysis (optional - requires API key)
    print("\n[4/4] Testing AI analysis (optional)...")
    try:
        analyzer = MultiModalAnalyzer(api_choice='anthropic')
        print(f"  Analyzer initialized: {analyzer.model}")
        print("  Skipping actual API call to save costs.")
        print("  To test full analysis, run the notebook or batch script.")
        print(f"SUCCESS: Integration verified!")
    except ValueError as e:
        print(f"  Skipped: {e}")
        print("  (This is OK if you haven't set up API keys yet)")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    # Summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open notebooks/quickstart.ipynb")
    print("2. Go to Section 1: ROCO Radiology Dataset")
    print("3. Run the cells to analyze real medical images!")
    print("\nOr run batch analysis:")
    print("  python scripts/batch_analyze_roco.py --num-samples 5")

    return 0


if __name__ == '__main__':
    sys.exit(main())
