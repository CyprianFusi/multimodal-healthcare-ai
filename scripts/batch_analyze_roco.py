"""
Batch Analysis of ROCO Dataset Images

This script processes multiple images from the ROCO dataset and generates
AI-powered radiology reports using Claude or GPT-4 Vision.

Usage:
    python scripts/batch_analyze_roco.py --num-samples 10 --split val --output results.csv
    python scripts/batch_analyze_roco.py --modality "X-ray" --num-samples 5
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal.processors.roco_loader import ROCODataset
from src.multimodal.analyzers.multimodal_analyzer import MultiModalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def batch_analyze(
    dataset_path: str = './data/roco',
    split: str = 'val',
    num_samples: int = 10,
    modality: str = None,
    api_choice: str = 'anthropic',
    output_file: str = None,
    save_images: bool = False
) -> List[Dict]:
    """
    Batch analyze ROCO images

    Args:
        dataset_path: Path to ROCO dataset
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to analyze
        modality: Optional modality filter
        api_choice: 'anthropic' or 'openai'
        output_file: Optional CSV file to save results
        save_images: Whether to save analyzed images

    Returns:
        List of analysis results
    """
    # Load dataset
    logger.info(f"Loading ROCO dataset from {dataset_path}...")
    try:
        roco = ROCODataset(dataset_path)
    except FileNotFoundError:
        logger.error(f"ROCO dataset not found at {dataset_path}")
        logger.info("Download it first using: python scripts/download_roco.py")
        return []

    # Initialize analyzer
    logger.info(f"Initializing analyzer with {api_choice}...")
    try:
        analyzer = MultiModalAnalyzer(api_choice=api_choice)
    except ValueError as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        logger.info("Check your API keys in .env file")
        return []

    # Process samples
    results = []
    logger.info(f"\nProcessing {num_samples} samples from {split} split...")
    logger.info("="*70)

    for i in range(num_samples):
        try:
            # Get sample
            if modality:
                sample = roco.get_random_sample(split=split, modality=modality)
            else:
                sample = roco.get_sample(split=split, index=i)

            logger.info(f"\n[{i+1}/{num_samples}] Processing: {sample['metadata']['image_id']}")
            logger.info(f"Modality: {sample['metadata'].get('modality', 'Unknown')}")

            # Prepare for analysis
            prepared = roco.prepare_for_analysis(sample)

            # Run AI analysis
            analysis_result = analyzer.analyze_medical_image(
                prepared,
                modality=sample['metadata'].get('modality', 'Unknown'),
                focus=None
            )

            # Compile results
            result = {
                'image_id': sample['metadata']['image_id'],
                'modality': sample['metadata'].get('modality', 'Unknown'),
                'ground_truth_caption': sample['caption'],
                'ai_analysis': analysis_result['analysis'],
                'latency': analysis_result['latency'],
                'input_tokens': analysis_result['tokens']['input'],
                'output_tokens': analysis_result['tokens']['output'],
                'split': split,
                'api_provider': api_choice,
                'model': analyzer.model,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

            logger.info(f"✓ Analysis complete ({analysis_result['latency']:.2f}s)")
            logger.info(f"  Tokens: {result['input_tokens']:,} in / {result['output_tokens']:,} out")

        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            continue

    # Get cost summary
    cost_summary = analyzer.get_cost_summary()
    logger.info("\n" + "="*70)
    logger.info("Batch Processing Complete!")
    logger.info("="*70)
    logger.info(f"Total samples processed: {len(results)}")
    logger.info(f"Total cost: ${cost_summary['total_cost']:.4f}")
    logger.info(f"Total tokens: {cost_summary['total_input_tokens']:,} in / {cost_summary['total_output_tokens']:,} out")
    logger.info(f"Average latency: {cost_summary['average_latency']:.2f}s")

    # Save results
    if output_file and results:
        save_results(results, output_file)

    return results


def save_results(results: List[Dict], output_file: str):
    """Save results to CSV file"""
    try:
        df = pd.DataFrame(results)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Results saved to: {output_path}")

        # Also save as JSON for better readability of long text
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Results also saved as JSON: {json_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch analyze ROCO dataset images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./data/roco',
        help='Path to ROCO dataset (default: ./data/roco)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='val',
        help='Dataset split to use (default: val)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to analyze (default: 10)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default=None,
        help='Filter by modality (e.g., "X-ray", "CT", "MRI")'
    )
    parser.add_argument(
        '--api-choice',
        type=str,
        choices=['anthropic', 'openai'],
        default='anthropic',
        help='API provider to use (default: anthropic)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (CSV format)'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Save analyzed images to disk'
    )

    args = parser.parse_args()

    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'./data/results/roco_analysis_{args.split}_{timestamp}.csv'

    logger.info("="*70)
    logger.info("ROCO Dataset Batch Analyzer")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Samples: {args.num_samples}")
    logger.info(f"  Modality filter: {args.modality or 'None'}")
    logger.info(f"  API: {args.api_choice}")
    logger.info(f"  Output: {args.output}")

    # Run batch analysis
    results = batch_analyze(
        dataset_path=args.dataset_path,
        split=args.split,
        num_samples=args.num_samples,
        modality=args.modality,
        api_choice=args.api_choice,
        output_file=args.output,
        save_images=args.save_images
    )

    if results:
        logger.info(f"\n✓ Successfully analyzed {len(results)} images")
        return 0
    else:
        logger.error("\n✗ No results generated")
        return 1


if __name__ == '__main__':
    sys.exit(main())
