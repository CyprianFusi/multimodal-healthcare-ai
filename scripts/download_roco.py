import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured"""
    kaggle_json_paths = [
        Path.home() / '.kaggle' / 'kaggle.json',  # Linux/Mac
        Path(f'C:/Users/{Path.home().name}/.kaggle/kaggle.json'),  # Windows
    ]

    for path in kaggle_json_paths:
        if path.exists():
            logger.info(f"Found Kaggle credentials at: {path}")
            return True

    return False


def download_roco(output_path: str = './data/roco') -> bool:
    """
    Download ROCO dataset from Kaggle

    Args:
        output_path: Directory to save the dataset

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if kaggle is installed
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.error("Kaggle package not installed!")
            logger.info("Install with: pip install kaggle")
            return False

        # Check credentials
        if not check_kaggle_credentials():
            logger.error("Kaggle API credentials not found!")
            logger.info("\nSetup instructions:")
            logger.info("1. Create account at https://kaggle.com")
            logger.info("2. Go to Account → API → Create New Token")
            logger.info("3. This downloads kaggle.json")
            logger.info("4. Place it in:")
            logger.info("   - Linux/Mac: ~/.kaggle/kaggle.json")
            logger.info("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
            logger.info("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
            return False

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Kaggle API
        logger.info("Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        logger.info("Authentication successful!")

        # Download dataset
        logger.info(f"Downloading ROCO dataset to: {output_dir}")
        logger.info("This may take a while (dataset is ~7.1 GB)...")

        api.dataset_download_files(
            'shareef0612/roco-radiology',
            path=str(output_dir),
            unzip=True
        )

        logger.info("Download complete!")

        # Verify download
        images_dir = output_dir / 'images'
        csv_files = list(output_dir.glob('*.csv'))

        if images_dir.exists() and len(csv_files) > 0:
            logger.info(f"\n✓ Dataset verified:")
            logger.info(f"  - Images directory: {images_dir}")
            logger.info(f"  - CSV files found: {len(csv_files)}")
            for csv_file in csv_files:
                logger.info(f"    - {csv_file.name}")

            # Count images
            image_count = len(list(images_dir.glob('*')))
            logger.info(f"  - Total images: {image_count:,}")

            return True
        else:
            logger.warning("Dataset downloaded but verification failed")
            logger.warning(f"Expected structure:")
            logger.warning(f"  {output_dir}/")
            logger.warning(f"  ├── images/")
            logger.warning(f"  ├── train_data.csv")
            logger.warning(f"  ├── val_data.csv")
            logger.warning(f"  └── test_data.csv")
            return False

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("\nAlternative: Manual download")
        logger.info("1. Visit: https://www.kaggle.com/datasets/shareef0612/roco-radiology")
        logger.info("2. Click 'Download' button")
        logger.info(f"3. Extract to: {output_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ROCO Radiology Dataset from Kaggle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='./data/roco',
        help='Directory to save the dataset (default: ./data/roco)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if credentials are configured, do not download'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("ROCO Radiology Dataset Downloader")
    logger.info("="*70)

    if args.check_only:
        if check_kaggle_credentials():
            logger.info("✓ Kaggle credentials are configured!")
            return 0
        else:
            logger.error("✗ Kaggle credentials not found")
            return 1

    # Download dataset
    success = download_roco(args.output_path)

    if success:
        logger.info("\n" + "="*70)
        logger.info("SUCCESS! Dataset ready to use")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Open notebooks/quickstart.ipynb")
        logger.info("2. Use ROCO examples instead of synthetic data")
        logger.info("3. Run analysis on real medical images!")
        return 0
    else:
        logger.error("\nDownload failed. See errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
