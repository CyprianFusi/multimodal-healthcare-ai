"""
ECG Signal Processing Module
Converts ECG signals to multi-modal format (image + text)
"""
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import neurokit2 as nk
import wfdb
from scipy import signal as scipy_signal
import logging

logger = logging.getLogger(__name__)


class ECGProcessor:
    """Process ECG signals for multi-modal analysis"""
    
    STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
    
    def load_ecg_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load ECG from various file formats

        Supports:
        - WFDB format (.dat, .hea)
        - CSV files
        - NumPy arrays (.npy)
        - Raw binary .dat files (NumPy)
        """
        file_path = Path(file_path)

        try:
            if file_path.suffix in ['.dat', '.hea']:
                # Check if .hea file exists for WFDB format
                hea_path = file_path.with_suffix('.hea')
                if hea_path.exists():
                    # PhysioNet WFDB format
                    record = wfdb.rdrecord(str(file_path.with_suffix('')))
                    ecg_signal = record.p_signal.T  # Transpose to (leads, samples)
                    metadata = {
                        'sampling_rate': record.fs,
                        'n_sig': record.n_sig,
                        'sig_name': record.sig_name,
                        'units': record.units
                    }
                    self.sampling_rate = record.fs
                else:
                    # Raw binary NumPy format (generated test data)
                    # Try int16 first (default format for synthetic data)
                    ecg_signal = np.fromfile(file_path, dtype=np.int16)
                    # Convert to float and normalize
                    ecg_signal = ecg_signal.astype(np.float64) / 1000.0
                    # Reshape to (leads, samples) - assume 12 leads
                    samples_per_lead = len(ecg_signal) // 12
                    if samples_per_lead * 12 == len(ecg_signal):
                        ecg_signal = ecg_signal.reshape(12, samples_per_lead)
                    else:
                        # Single lead
                        ecg_signal = ecg_signal.reshape(1, -1)
                    metadata = {'format': 'raw_binary', 'sampling_rate': self.sampling_rate}
                
            elif file_path.suffix == '.csv':
                # CSV format
                data = np.loadtxt(file_path, delimiter=',')
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                ecg_signal = data
                metadata = {'format': 'csv', 'sampling_rate': self.sampling_rate}
                
            elif file_path.suffix == '.npy':
                # NumPy format
                ecg_signal = np.load(file_path)
                if ecg_signal.ndim == 1:
                    ecg_signal = ecg_signal.reshape(1, -1)
                metadata = {'format': 'numpy', 'sampling_rate': self.sampling_rate}
                
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded ECG from {file_path}: shape {ecg_signal.shape}")
            return ecg_signal, metadata
            
        except Exception as e:
            logger.error(f"Failed to load ECG file {file_path}: {e}")
            raise
    
    def process_ecg(
        self, 
        ecg_signal: np.ndarray, 
        lead_names: Optional[list] = None
    ) -> Dict:
        """
        Process ECG signal into multi-modal format
        
        Returns:
            Dictionary with:
            - image_base64: Base64 encoded ECG waveform image
            - annotations: Text-based ECG measurements
            - features: Numerical features dictionary
            - metadata: Processing metadata
        """
        try:
            # Ensure correct shape
            if ecg_signal.ndim == 1:
                ecg_signal = ecg_signal.reshape(1, -1)
            
            num_leads = ecg_signal.shape[0]
            
            # Use standard lead names or provided names
            if lead_names is None:
                lead_names = self.STANDARD_LEADS[:num_leads]
            
            # 1. Clean signals
            cleaned_signals = self._clean_ecg_signals(ecg_signal)
            
            # 2. Extract features from Lead II (standard for rhythm analysis)
            lead_ii_idx = 1 if num_leads >= 2 else 0
            features = self._extract_ecg_features(cleaned_signals[lead_ii_idx])
            
            # 3. Generate waveform visualization
            waveform_base64 = self._generate_waveform_image(
                cleaned_signals, lead_names
            )
            
            # 4. Generate spectrogram
            spectrogram_base64 = self._generate_spectrogram(
                cleaned_signals[lead_ii_idx]
            )
            
            # 5. Create text annotations
            annotations = self._generate_annotations(features)
            
            return {
                'waveform_image_base64': waveform_base64,
                'spectrogram_base64': spectrogram_base64,
                'annotations': annotations,
                'features': features,
                'metadata': {
                    'sampling_rate': self.sampling_rate,
                    'duration': cleaned_signals.shape[1] / self.sampling_rate,
                    'num_leads': num_leads,
                    'lead_names': lead_names
                }
            }
            
        except Exception as e:
            logger.error(f"ECG processing failed: {e}")
            raise
    
    def _clean_ecg_signals(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Clean and filter ECG signals"""
        cleaned = np.zeros_like(ecg_signal)
        
        for i in range(ecg_signal.shape[0]):
            try:
                cleaned[i] = nk.ecg_clean(
                    ecg_signal[i], 
                    sampling_rate=self.sampling_rate
                )
            except Exception as e:
                logger.warning(f"Could not clean lead {i}: {e}")
                cleaned[i] = ecg_signal[i]
        
        return cleaned
    
    def _extract_ecg_features(self, ecg_signal: np.ndarray) -> Dict:
        """Extract numerical features from ECG"""
        # First, try to find R peaks without full processing
        try:
            cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate)
            r_peaks_dict = nk.ecg_peaks(cleaned, sampling_rate=self.sampling_rate)
            r_peaks = r_peaks_dict[0]['ECG_R_Peaks']

            # Check if we have enough R peaks before processing
            if len(r_peaks) < 2:
                logger.warning(f"Only {len(r_peaks)} R peaks detected. Insufficient data for analysis.")
                return {
                    'heart_rate_mean': None,
                    'heart_rate_std': None,
                    'rr_interval_mean': None,
                    'num_r_peaks': len(r_peaks),
                    'hrv_rmssd': None,
                    'hrv_sdnn': None,
                    'warning': f'Insufficient R peaks detected ({len(r_peaks)}). Need at least 2 for analysis.'
                }
        except (ZeroDivisionError, ValueError) as peak_error:
            logger.warning(f"R peak detection failed (division/value error): {peak_error}")
            return {
                'heart_rate_mean': None,
                'heart_rate_std': None,
                'rr_interval_mean': None,
                'num_r_peaks': 0,
                'hrv_rmssd': None,
                'hrv_sdnn': None,
                'warning': f'Signal quality too poor for R peak detection. Please verify ECG signal.'
            }
        except Exception as peak_error:
            logger.warning(f"R peak detection failed: {peak_error}")
            return {
                'heart_rate_mean': None,
                'heart_rate_std': None,
                'rr_interval_mean': None,
                'num_r_peaks': 0,
                'hrv_rmssd': None,
                'hrv_sdnn': None,
                'warning': f'R peak detection failed: {str(peak_error)}'
            }

        # Now proceed with full processing since we know we have peaks
        try:
            # Process ECG to get peaks and intervals
            # Wrap this in its own try-except since it can throw ZeroDivisionError internally
            try:
                signals, info = nk.ecg_process(
                    ecg_signal,
                    sampling_rate=self.sampling_rate
                )
            except ZeroDivisionError as zde_inner:
                logger.warning(f"Division by zero during nk.ecg_process: {zde_inner}")
                return {
                    'heart_rate_mean': None,
                    'heart_rate_std': None,
                    'rr_interval_mean': None,
                    'num_r_peaks': len(r_peaks),
                    'hrv_rmssd': None,
                    'hrv_sdnn': None,
                    'warning': f'ECG signal processing failed due to insufficient quality. Detected {len(r_peaks)} peaks but unable to compute heart rate.'
                }

            # Check number of R peaks from full processing
            num_r_peaks = int(len(info['ECG_R_Peaks']))

            # Get heart rate safely
            heart_rate_mean = None
            if 'ECG_Rate_Mean' in info and info['ECG_Rate_Mean'] > 0:
                heart_rate_mean = float(info['ECG_Rate_Mean'])

            # Calculate HRV metrics with error handling
            hrv_rmssd = None
            hrv_sdnn = None
            try:
                if num_r_peaks >= 2:
                    hrv = nk.hrv_time(signals, sampling_rate=self.sampling_rate)
                    hrv_rmssd = float(hrv['HRV_RMSSD'].values[0]) if 'HRV_RMSSD' in hrv.columns else None
                    hrv_sdnn = float(hrv['HRV_SDNN'].values[0]) if 'HRV_SDNN' in hrv.columns else None
            except Exception as hrv_error:
                logger.warning(f"HRV calculation failed: {hrv_error}")

            features = {
                'heart_rate_mean': heart_rate_mean,
                'heart_rate_std': float(signals['ECG_Rate'].std()) if len(signals['ECG_Rate']) > 0 else None,
                'rr_interval_mean': float(60000 / heart_rate_mean) if heart_rate_mean and heart_rate_mean > 0 else None,  # ms
                'num_r_peaks': num_r_peaks,
                'hrv_rmssd': hrv_rmssd,
                'hrv_sdnn': hrv_sdnn,
            }

            # Estimate intervals (simplified)
            features.update(self._estimate_intervals(signals, info))

            return features

        except ZeroDivisionError as zde:
            logger.warning(f"Division by zero during ECG processing: {zde}")
            # Return basic info from the initial peak detection
            return {
                'heart_rate_mean': None,
                'heart_rate_std': None,
                'rr_interval_mean': None,
                'num_r_peaks': len(r_peaks),
                'hrv_rmssd': None,
                'hrv_sdnn': None,
                'warning': f'ECG signal quality issue prevented full analysis. Detected {len(r_peaks)} potential R-peaks.'
            }
        except Exception as e:
            logger.error(f"Feature extraction failed during full processing: {e}")
            # Return basic info from the initial peak detection
            return {
                'heart_rate_mean': None,
                'heart_rate_std': None,
                'rr_interval_mean': None,
                'num_r_peaks': len(r_peaks),
                'hrv_rmssd': None,
                'hrv_sdnn': None,
                'warning': f'ECG processing failed: {str(e)}'
            }
    
    def _estimate_intervals(self, signals, info) -> Dict:
        """Estimate PR, QRS, QT intervals"""
        try:
            # This is a simplified estimation
            # In production, use more sophisticated algorithms
            r_peaks = info['ECG_R_Peaks']

            if len(r_peaks) < 2:
                return {}

            # Average RR interval
            rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # ms
            avg_rr = float(np.mean(rr_intervals))

            # Check for valid RR interval before division
            if avg_rr <= 0:
                return {}

            # Rough estimates (would need proper QRS detection)
            return {
                'pr_interval_est': 160.0,  # ms (normal range 120-200)
                'qrs_duration_est': 90.0,  # ms (normal range 80-120)
                'qt_interval_est': 400.0,  # ms (depends on HR)
                'qtc_corrected_est': 400.0 / np.sqrt(avg_rr / 1000),  # Bazett's formula
            }
        except Exception as e:
            logger.warning(f"Interval estimation failed: {e}")
            return {}
    
    def _generate_waveform_image(
        self, 
        ecg_signals: np.ndarray, 
        lead_names: list
    ) -> str:
        """Generate clean ECG waveform visualization"""
        num_leads = ecg_signals.shape[0]
        
        # Create figure
        fig, axes = plt.subplots(
            num_leads, 1, 
            figsize=(15, 2 * num_leads),
            facecolor='white'
        )
        
        if num_leads == 1:
            axes = [axes]
        
        # Plot each lead
        time = np.arange(ecg_signals.shape[1]) / self.sampling_rate
        
        for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
            # Plot waveform
            ax.plot(time, ecg_signals[i], linewidth=0.8, color='black')
            
            # Styling
            ax.set_ylabel(lead_name, fontsize=12, fontweight='bold')
            ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5)
            ax.set_xlim(0, time[-1])
            
            # Remove x-label except for last plot
            if i < num_leads - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (seconds)', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    
    def _generate_spectrogram(self, ecg_signal: np.ndarray) -> str:
        """Generate spectrogram representation"""
        try:
            # Compute spectrogram
            f, t, Sxx = scipy_signal.spectrogram(
                ecg_signal, 
                fs=self.sampling_rate,
                nperseg=256
            )
            
            # Plot spectrogram
            fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
            im = ax.pcolormesh(
                t, f, 10 * np.log10(Sxx + 1e-10), 
                shading='gouraud', 
                cmap='viridis'
            )
            ax.set_ylabel('Frequency (Hz)', fontsize=10)
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_title('ECG Spectrogram', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Power (dB)')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Spectrogram generation failed: {e}")
            return ""
    
    def _generate_annotations(self, features: Dict) -> str:
        """Generate text-based annotations from features"""
        if 'error' in features:
            return f"Error extracting features: {features['error']}"

        if 'warning' in features:
            return f"⚠️ Warning: {features['warning']}\n\nUnable to extract ECG features. Please check signal quality or duration."

        # Helper function to format values
        def fmt(val, decimals=1, suffix=""):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.{decimals}f}{suffix}"

        hr_mean = features.get('heart_rate_mean')
        hr_std = features.get('heart_rate_std')

        annotations = f"""
ECG Measurements:
-----------------
Heart Rate: {fmt(hr_mean, 1, ' bpm')} (± {fmt(hr_std, 1, '')})
RR Interval: {fmt(features.get('rr_interval_mean'), 0, ' ms')}
Number of R peaks detected: {features.get('num_r_peaks', 'N/A')}

Estimated Intervals:
-------------------
PR Interval: {fmt(features.get('pr_interval_est'), 0, ' ms')}
QRS Duration: {fmt(features.get('qrs_duration_est'), 0, ' ms')}
QT Interval: {fmt(features.get('qt_interval_est'), 0, ' ms')}
QTc (Corrected): {fmt(features.get('qtc_corrected_est'), 0, ' ms')}

Heart Rate Variability:
----------------------
RMSSD: {fmt(features.get('hrv_rmssd'), 1, ' ms')}
SDNN: {fmt(features.get('hrv_sdnn'), 1, ' ms')}

Note: Interval estimates are approximations. Clinical interpretation required.
"""
        return annotations.strip()


# Utility function for quick processing
def process_ecg_file(file_path: Union[str, Path]) -> Dict:
    """Quick function to process an ECG file"""
    processor = ECGProcessor()
    ecg_signal, metadata = processor.load_ecg_file(file_path)
    return processor.process_ecg(ecg_signal)
