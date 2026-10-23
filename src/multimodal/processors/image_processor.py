"""
Medical Image Processing Module
Handles DICOM, X-rays, CT scans, MRI and other medical imaging
"""
import numpy as np
import pydicom
from PIL import Image
import cv2
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MedicalImageProcessor:
    """Process medical images for multi-modal analysis"""

    SUPPORTED_MODALITIES = ['CR', 'CT', 'MR', 'US', 'XA', 'DX', 'MG', 'PT']

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def load_image(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load medical image from various formats
        
        Supports:
        - DICOM (.dcm)
        - Standard images (.png, .jpg, .jpeg)
        - NIfTI (.nii, .nii.gz) - basic support
        """
        file_path = Path(file_path)

        try:
            if file_path.suffix.lower() == '.dcm':
                return self._load_dicom(file_path)
            elif file_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                return self._load_standard_image(file_path)
            elif file_path.suffix.lower() in ('.nii', '.gz'):
                return self._load_nifti(file_path)
            else:
                raise ValueError(f"Unsupported image format: {file_path.suffix}")

        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise

    def _load_dicom(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load DICOM image and extract metadata"""
        try:
            dcm = pydicom.dcmread(file_path)

            # Extract pixel data
            image_array = dcm.pixel_array

            # Extract relevant metadata
            metadata = {
                'modality': str(getattr(dcm, 'Modality', 'Unknown')),
                'body_part': str(getattr(dcm, 'BodyPartExamined', 'N/A')),
                'study_date': str(getattr(dcm, 'StudyDate', 'N/A')),
                'study_description': str(getattr(dcm, 'StudyDescription', 'N/A')),
                'series_description': str(getattr(dcm, 'SeriesDescription', 'N/A')),
                'patient_age': str(getattr(dcm, 'PatientAge', 'N/A')),
                'patient_sex': str(getattr(dcm, 'PatientSex', 'N/A')),
                'pixel_spacing': list(getattr(dcm, 'PixelSpacing', [])) if hasattr(dcm, 'PixelSpacing') else None,
                'slice_thickness': float(getattr(dcm, 'SliceThickness', 0)) if hasattr(dcm, 'SliceThickness') else None,
                'window_center': float(getattr(dcm, 'WindowCenter', 0)) if hasattr(dcm, 'WindowCenter') else None,
                'window_width': float(getattr(dcm, 'WindowWidth', 0)) if hasattr(dcm, 'WindowWidth') else None,
                'photometric_interpretation': str(getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')),
                'rows': int(dcm.Rows),
                'columns': int(dcm.Columns),
                'format': 'DICOM'
            }

            logger.info(f"Loaded DICOM: {metadata['modality']} - {metadata['body_part']}")
            return image_array, metadata

        except Exception as e:
            logger.error(f"Failed to load DICOM {file_path}: {e}")
            raise

    def _load_standard_image(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load standard image formats"""
        try:
            image = Image.open(file_path)
            image_array = np.array(image)

            metadata = {
                'format': file_path.suffix.upper().replace('.', ''),
                'size': image.size,
                'mode': image.mode
            }

            # Convert to grayscale if RGB
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            return image_array, metadata

        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise

    def _load_nifti(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Basic NIfTI support (requires nibabel)"""
        try:
            import nibabel as nib

            nii = nib.load(file_path)
            image_array = nii.get_fdata()

            # Take middle slice for 3D volumes
            if image_array.ndim == 3:
                image_array = image_array[:, :, image_array.shape[2] // 2]

            metadata = {
                'format': 'NIfTI',
                'shape': image_array.shape,
                'voxel_dims': nii.header.get_zooms()
            }

            return image_array, metadata

        except ImportError:
            logger.error("nibabel not installed. Cannot load NIfTI files.")
            raise
        except Exception as e:
            logger.error(f"Failed to load NIfTI {file_path}: {e}")
            raise

    def process_image(
        self,
        image_array: np.ndarray,
        metadata: Dict,
        anonymize: bool = True
    ) -> Dict:
        """
        Process medical image for API consumption
        
        Returns:
            Dictionary with:
            - image_base64: Base64 encoded processed image
            - metadata_text: Formatted metadata as text
            - metadata: Original metadata dict
        """
        try:
            # Anonymize metadata if requested
            if anonymize:
                metadata = self._anonymize_metadata(metadata)

            # Normalize intensities
            processed_image = self._normalize_image(image_array, metadata)

            # Enhance contrast
            enhanced_image = self._enhance_image(processed_image)

            # Convert to RGB PIL Image
            image_rgb = self._to_rgb_image(enhanced_image)

            # Resize to reasonable dimensions
            if max(image_rgb.size) > 2048:
                image_rgb.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            elif min(image_rgb.size) < self.target_size[0]:
                image_rgb = image_rgb.resize(self.target_size, Image.Resampling.LANCZOS)

            # Encode for transport
            image_base64 = self._image_to_base64(image_rgb)

            # Format metadata for the language model
            metadata_text = self._format_metadata(metadata)

            return {
                'image_base64': image_base64,
                'metadata_text': metadata_text,
                'metadata': metadata,
                'processed_size': image_rgb.size
            }

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise

    def _anonymize_metadata(self, metadata: Dict) -> Dict:
        """Remove PHI from metadata"""
        anonymized = metadata.copy()

        # Remove or mask identifying fields
        phi_fields = ['PatientName', 'PatientID', 'PatientBirthDate']
        for field in phi_fields:
            if field in anonymized:
                anonymized[field] = 'ANONYMIZED'

        return anonymized

    def _normalize_image(self, image_array: np.ndarray, metadata: Dict) -> np.ndarray:
        """Normalize image intensities"""
        try:
            # Apply DICOM windowing if available
            if metadata.get('window_center') and metadata.get('window_width'):
                center = float(metadata['window_center'])
                width = float(metadata['window_width'])

                lower = center - width / 2
                upper = center + width / 2

                image_array = np.clip(image_array, lower, upper)

            # Scale to 8-bit range
            if image_array.dtype != np.uint8:
                image_min = image_array.min()
                image_max = image_array.max()

                if image_max > image_min:
                    image_array = ((image_array - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)

            return image_array

        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return image_array

    def _enhance_image(self, image_array: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better contrast"""
        try:
            # CLAHE requires 8-bit input
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)

            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image_array)

            return enhanced

        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return image_array

    def _to_rgb_image(self, image_array: np.ndarray) -> Image.Image:
        """Convert numpy array to RGB PIL Image"""
        if image_array.ndim == 2:
            # Grayscale to RGB
            image_rgb = Image.fromarray(image_array).convert('RGB')
        else:
            image_rgb = Image.fromarray(image_array)

        return image_rgb

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    def _format_metadata(self, metadata: Dict) -> str:
        """Format metadata as readable text"""
        lines = ["Imaging Metadata:", "=" * 50]

        # Field name to display label mapping
        display_fields = [
            ('modality', 'Modality'),
            ('body_part', 'Body Part Examined'),
            ('study_date', 'Study Date'),
            ('study_description', 'Study Description'),
            ('series_description', 'Series Description'),
            ('patient_age', 'Patient Age'),
            ('patient_sex', 'Patient Sex'),
            ('pixel_spacing', 'Pixel Spacing (mm)'),
            ('slice_thickness', 'Slice Thickness (mm)'),
            ('rows', 'Image Rows'),
            ('columns', 'Image Columns'),
            ('format', 'Image Format')
        ]

        for key, label in display_fields:
            if key in metadata and metadata[key] not in ('N/A', None, '', 'Unknown'):
                value = metadata[key]
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    def create_thumbnail(self, image_array: np.ndarray, size: Tuple[int, int] = (128, 128)) -> str:
        """Create thumbnail for preview"""
        try:
            # Normalize without windowing metadata
            normalized = self._normalize_image(image_array, {})

            # Wrap in PIL for resampling
            image = Image.fromarray(normalized)

            # Preserve aspect ratio
            image.thumbnail(size, Image.Resampling.LANCZOS)

            # Encode for transport
            return self._image_to_base64(image)

        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            return ""

    def process_dicom_series(self, dicom_dir: Path) -> Dict:
        """Process entire DICOM series (for CT/MRI)"""
        try:
            dicom_files = sorted(dicom_dir.glob('*.dcm'))

            if not dicom_files:
                raise ValueError(f"No DICOM files found in {dicom_dir}")

            # Load every slice in the series
            slices = []
            metadata_list = []

            for dcm_file in dicom_files:
                image_array, metadata = self._load_dicom(dcm_file)
                slices.append(image_array)
                metadata_list.append(metadata)

            # Build 3D volume
            volume = np.stack(slices, axis=0)

            # Representative slice for analysis
            middle_idx = volume.shape[0] // 2
            middle_slice = volume[middle_idx]

            # Process the representative slice
            result = self.process_image(middle_slice, metadata_list[middle_idx])
            result['series_info'] = {
                'num_slices': len(slices),
                'volume_shape': volume.shape,
                'middle_slice_idx': middle_idx
            }

            return result

        except Exception as e:
            logger.error(f"DICOM series processing failed: {e}")
            raise


# Convenience function
def process_medical_image_file(file_path: Union[str, Path]) -> Dict:
    """Quick function to process a medical image file"""
    processor = MedicalImageProcessor()
    image_array, metadata = processor.load_image(file_path)
    return processor.process_image(image_array, metadata)
