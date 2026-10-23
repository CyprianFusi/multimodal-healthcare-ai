"""
ROCO Dataset Loader
Loads and processes images from the ROCO Radiology dataset
https://www.kaggle.com/datasets/shareef0612/roco-radiology
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from PIL import Image
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class ROCODataset:
    """Loader for ROCO Radiology dataset"""

    def __init__(self, dataset_path: str):
        """
        Initialize ROCO dataset loader

        Args:
            dataset_path: Path to ROCO dataset directory containing images/ and CSV files
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'images'

        # Check if dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"ROCO dataset not found at {dataset_path}\n"
                "Please download from: https://www.kaggle.com/datasets/shareef0612/roco-radiology"
            )

        # Load metadata CSVs
        self.train_df = self._load_csv('train_data.csv')
        self.val_df = self._load_csv('val_data.csv')
        self.test_df = self._load_csv('test_data.csv')

        logger.info(f"ROCO dataset loaded from {dataset_path}")
        logger.info(f"Train: {len(self.train_df)} samples")
        logger.info(f"Val: {len(self.val_df)} samples")
        logger.info(f"Test: {len(self.test_df)} samples")

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file if it exists"""
        csv_path = self.dataset_path / filename
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            logger.warning(f"{filename} not found, returning empty DataFrame")
            return pd.DataFrame()

    def get_sample(self, split: str = 'train', index: int = 0) -> Dict:
        """
        Get a single sample from the dataset

        Args:
            split: 'train', 'val', or 'test'
            index: Sample index

        Returns:
            Dict with 'image', 'image_path', 'caption', 'metadata'
        """
        # Select appropriate dataframe
        if split == 'train':
            df = self.train_df
        elif split == 'val':
            df = self.val_df
        elif split == 'test':
            df = self.test_df
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")

        if index >= len(df):
            raise IndexError(f"Index {index} out of range for {split} split (size: {len(df)})")

        # Get row
        row = df.iloc[index]

        # Load image - ROCO uses PMC_ID column for filename
        # Images are organized in split subdirectories: train_set, val_set, test_set
        image_filename = row.get('PMC_ID', row.get('name', row.get('image_id', f'{index}.jpg')))

        # Construct path with split subdirectory
        split_dir = f'{split}_set'
        image_path = self.images_dir / split_dir / image_filename

        if not image_path.exists():
            # Try without split subdirectory (fallback)
            image_path = self.images_dir / image_filename

            if not image_path.exists():
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = image_path.with_suffix(ext)
                    if alt_path.exists():
                        image_path = alt_path
                        break

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return placeholder
            image = Image.new('RGB', (512, 512), color='black')

        # Extract metadata - ROCO uses capitalized column names
        caption = row.get('Caption', row.get('caption', ''))
        keywords = row.get('Keywords', row.get('keywords', ''))

        metadata = {
            'image_id': image_filename,
            'roco_id': row.get('ROCO_ID', ''),
            'caption': caption,
            'keywords': keywords,
            'modality': row.get('modality', 'Unknown'),  # May not be in CSV
            'split': split,
            'index': index
        }

        # Add all other columns as metadata
        for col in df.columns:
            if col not in ['PMC_ID', 'Caption', 'Keywords', 'ROCO_ID', 'name', 'image_id', 'caption', 'modality', 'keywords']:
                metadata[col] = row.get(col, None)

        return {
            'image': image,
            'image_path': str(image_path),
            'caption': metadata['caption'],
            'metadata': metadata
        }

    def get_random_sample(self, split: str = 'train', modality: Optional[str] = None) -> Dict:
        """
        Get a random sample from the dataset

        Args:
            split: 'train', 'val', or 'test'
            modality: Optional filter by modality (e.g., 'X-ray', 'CT', 'MRI')

        Returns:
            Dict with sample data
        """
        # Select appropriate dataframe
        if split == 'train':
            df = self.train_df
        elif split == 'val':
            df = self.val_df
        elif split == 'test':
            df = self.test_df
        else:
            raise ValueError(f"Invalid split: {split}")

        # Filter by modality if specified
        if modality and 'modality' in df.columns:
            filtered_df = df[df['modality'].str.contains(modality, case=False, na=False)]
            if len(filtered_df) == 0:
                logger.warning(f"No samples found for modality '{modality}', using all samples")
            else:
                df = filtered_df

        # Use pandas sample() for better randomness (uses numpy's random state)
        sampled_row = df.sample(n=1, random_state=None)  # random_state=None uses fresh seed
        random_idx = sampled_row.index[0]

        # Get the position in the dataframe (for get_sample which uses iloc)
        position = df.index.get_loc(random_idx)

        return self.get_sample(split=split, index=position)

    def get_batch(self, split: str = 'train', start_idx: int = 0, batch_size: int = 10) -> List[Dict]:
        """
        Get a batch of samples

        Args:
            split: 'train', 'val', or 'test'
            start_idx: Starting index
            batch_size: Number of samples to retrieve

        Returns:
            List of sample dicts
        """
        samples = []
        for i in range(start_idx, min(start_idx + batch_size, self.get_split_size(split))):
            try:
                sample = self.get_sample(split=split, index=i)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load sample {i}: {e}")
                continue

        return samples

    def get_split_size(self, split: str) -> int:
        """Get size of a split"""
        if split == 'train':
            return len(self.train_df)
        elif split == 'val':
            return len(self.val_df)
        elif split == 'test':
            return len(self.test_df)
        else:
            raise ValueError(f"Invalid split: {split}")

    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string

        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def prepare_for_analysis(self, sample: Dict, target_size: Tuple[int, int] = (512, 512)) -> Dict:
        """
        Prepare a ROCO sample for medical image analysis

        Args:
            sample: Sample dict from get_sample()
            target_size: Resize target (width, height)

        Returns:
            Dict ready for MultiModalAnalyzer.analyze_medical_image()
        """
        image = sample['image']

        # Resize if needed
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to base64
        image_base64 = self.image_to_base64(image)

        # Format metadata for analysis
        metadata_text = f"""
ROCO ID: {sample['metadata'].get('roco_id', 'Unknown')}
Image ID: {sample['metadata']['image_id']}
Keywords: {sample['metadata'].get('keywords', 'N/A')}
Ground Truth Caption: {sample['caption']}
""".strip()

        return {
            'image_base64': image_base64,
            'metadata_text': metadata_text,
            'metadata': sample['metadata'],
            'ground_truth_caption': sample['caption']
        }

    def get_modality_distribution(self, split: str = 'train') -> Dict[str, int]:
        """Get distribution of imaging modalities in a split"""
        if split == 'train':
            df = self.train_df
        elif split == 'val':
            df = self.val_df
        elif split == 'test':
            df = self.test_df
        else:
            raise ValueError(f"Invalid split: {split}")

        if 'modality' in df.columns:
            return df['modality'].value_counts().to_dict()
        else:
            return {}

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_samples': len(self.train_df) + len(self.val_df) + len(self.test_df),
            'train_samples': len(self.train_df),
            'val_samples': len(self.val_df),
            'test_samples': len(self.test_df),
            'train_modalities': self.get_modality_distribution('train'),
            'val_modalities': self.get_modality_distribution('val'),
            'test_modalities': self.get_modality_distribution('test')
        }


def download_roco_dataset(output_path: str = './data/roco') -> None:
    """
    Download ROCO dataset from Kaggle using Kaggle API

    Args:
        output_path: Where to download the dataset

    Note:
        Requires kaggle package and API credentials configured
        Run: pip install kaggle
        Setup: https://github.com/Kaggle/kaggle-api#api-credentials
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize API
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        logger.info("Downloading ROCO dataset from Kaggle...")
        api.dataset_download_files(
            'shareef0612/roco-radiology',
            path=output_path,
            unzip=True
        )
        logger.info(f"Dataset downloaded to {output_path}")

    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("Manual download: https://www.kaggle.com/datasets/shareef0612/roco-radiology")
        raise


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load dataset
    try:
        roco = ROCODataset('./data/roco')

        # Print stats
        stats = roco.get_stats()
        print("\nROCO Dataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Train: {stats['train_samples']}")
        print(f"Val: {stats['val_samples']}")
        print(f"Test: {stats['test_samples']}")

        # Get random sample
        sample = roco.get_random_sample(split='train')
        print(f"\nRandom sample:")
        print(f"Image: {sample['image_path']}")
        print(f"Caption: {sample['caption'][:100]}...")

    except FileNotFoundError:
        print("ROCO dataset not found. Download it first:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("3. Run: python -c 'from src.multimodal.processors.roco_loader import download_roco_dataset; download_roco_dataset()'")
