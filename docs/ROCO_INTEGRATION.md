# ROCO Dataset Integration Guide

This guide explains how to use the ROCO Radiology dataset with your Multi-Modal Healthcare AI system.

## 📖 What is ROCO?

**ROCO (Radiology Objects in COntext)** is a large-scale multimodal medical imaging dataset containing:

- **81,800+ radiology images** from medical literature
- **Image-text pairs** with radiological captions
- **Multiple modalities**: X-ray, CT, MRI, Ultrasound
- **Real clinical images** (much better than synthetic data!)

**Dataset**: https://www.kaggle.com/datasets/shareef0612/roco-radiology

## 🚀 Quick Start

### 1. Download ROCO Dataset

**Option A: Automatic Download (Recommended)**

```bash
# Install kaggle package
pip install kaggle

# Setup Kaggle API credentials
# Visit: https://github.com/Kaggle/kaggle-api#api-credentials
# Download kaggle.json and place in:
#   - Linux/Mac: ~/.kaggle/kaggle.json
#   - Windows: C:\Users\<username>\.kaggle\kaggle.json

# Download ROCO dataset (~7.1 GB)
python scripts/download_roco.py
```

**Option B: Manual Download**

1. Visit https://www.kaggle.com/datasets/shareef0612/roco-radiology
2. Click "Download" button
3. Extract to `./data/roco/`

### 2. Verify Installation

```python
from src.multimodal.processors.roco_loader import ROCODataset

# Load dataset
roco = ROCODataset('./data/roco')

# Get statistics
stats = roco.get_stats()
print(f"Total samples: {stats['total_samples']:,}")
```

### 3. Analyze Images

**In Jupyter Notebook:**

```python
# Get random image
sample = roco.get_random_sample(split='val', modality='X-ray')

# Prepare for analysis
prepared = roco.prepare_for_analysis(sample)

# Analyze with AI
from src.multimodal.analyzers.multimodal_analyzer import MultiModalAnalyzer
analyzer = MultiModalAnalyzer(api_choice='anthropic')
result = analyzer.analyze_medical_image(prepared, modality='X-ray')

print(result['analysis'])
```

**Batch Processing:**

```bash
# Analyze 10 random validation images
python scripts/batch_analyze_roco.py --num-samples 10 --split val

# Analyze only X-ray images
python scripts/batch_analyze_roco.py --modality "X-ray" --num-samples 5

# Custom output file
python scripts/batch_analyze_roco.py --output my_results.csv --num-samples 20
```

## 📁 Dataset Structure

```
data/roco/
├── images/              # 81,800+ radiology images
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── train_data.csv       # Training split with metadata
├── val_data.csv         # Validation split
└── test_data.csv        # Test split
```

### CSV Columns

Each CSV contains ~12 columns with metadata:
- `name` or `image_id`: Image filename
- `caption`: Radiological description from medical literature
- `modality`: Imaging modality (X-ray, CT, MRI, etc.)
- Additional metadata fields

## 🔧 ROCODataset API Reference

### Initialize Dataset

```python
from src.multimodal.processors.roco_loader import ROCODataset

roco = ROCODataset('./data/roco')
```

### Get Samples

```python
# Get specific sample by index
sample = roco.get_sample(split='train', index=0)

# Get random sample
sample = roco.get_random_sample(split='val')

# Get random sample of specific modality
sample = roco.get_random_sample(split='val', modality='X-ray')

# Get batch of samples
batch = roco.get_batch(split='train', start_idx=0, batch_size=10)
```

### Sample Structure

```python
{
    'image': PIL.Image,           # PIL Image object
    'image_path': str,            # Path to image file
    'caption': str,               # Ground truth caption
    'metadata': {
        'image_id': str,
        'modality': str,
        'split': str,
        'index': int,
        # ... other CSV columns
    }
}
```

### Prepare for Analysis

```python
# Convert sample to format needed by MultiModalAnalyzer
prepared = roco.prepare_for_analysis(sample, target_size=(512, 512))

# Prepared structure:
{
    'image_base64': str,          # Base64 encoded image
    'metadata_text': str,         # Formatted metadata string
    'metadata': dict,             # Original metadata
    'ground_truth_caption': str   # Ground truth for comparison
}
```

### Dataset Statistics

```python
# Get statistics
stats = roco.get_stats()
print(f"Total samples: {stats['total_samples']}")
print(f"Train: {stats['train_samples']}")
print(f"Val: {stats['val_samples']}")
print(f"Test: {stats['test_samples']}")

# Get modality distribution
modalities = roco.get_modality_distribution('train')
for modality, count in modalities.items():
    print(f"{modality}: {count}")
```

## 💡 Usage Examples

### Example 1: Compare AI vs Ground Truth

```python
from src.multimodal.processors.roco_loader import ROCODataset
from src.multimodal.analyzers.multimodal_analyzer import MultiModalAnalyzer

# Load dataset
roco = ROCODataset('./data/roco')
analyzer = MultiModalAnalyzer(api_choice='anthropic')

# Get sample
sample = roco.get_random_sample(split='val', modality='X-ray')
prepared = roco.prepare_for_analysis(sample)

# Analyze
result = analyzer.analyze_medical_image(prepared, modality='X-ray')

# Compare
print("="*70)
print("AI-Generated Analysis:")
print("="*70)
print(result['analysis'])

print("\n" + "="*70)
print("Ground Truth Caption:")
print("="*70)
print(sample['caption'])
```

### Example 2: Batch Processing Different Modalities

```python
modalities = ['X-ray', 'CT', 'MRI']

for modality in modalities:
    print(f"\nProcessing {modality} images...")

    # Process 5 images per modality
    for i in range(5):
        sample = roco.get_random_sample(split='val', modality=modality)
        prepared = roco.prepare_for_analysis(sample)

        result = analyzer.analyze_medical_image(prepared, modality=modality)

        print(f"  {i+1}. {sample['metadata']['image_id']}")
        print(f"     Latency: {result['latency']:.2f}s")
```

### Example 3: Save Analysis Results

```python
import pandas as pd

results = []

for i in range(10):
    sample = roco.get_sample(split='val', index=i)
    prepared = roco.prepare_for_analysis(sample)

    analysis = analyzer.analyze_medical_image(prepared, modality='Unknown')

    results.append({
        'image_id': sample['metadata']['image_id'],
        'modality': sample['metadata'].get('modality'),
        'ground_truth': sample['caption'],
        'ai_analysis': analysis['analysis'],
        'latency': analysis['latency']
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('./results/roco_analysis.csv', index=False)
```

## 📊 Batch Analysis Script

The batch analysis script automates processing multiple images:

### Basic Usage

```bash
# Analyze 10 validation images
python scripts/batch_analyze_roco.py --num-samples 10

# Output:
# - CSV file: ./data/results/roco_analysis_val_<timestamp>.csv
# - JSON file: ./data/results/roco_analysis_val_<timestamp>.json
```

### Advanced Options

```bash
# Filter by modality
python scripts/batch_analyze_roco.py \
    --modality "X-ray" \
    --num-samples 20 \
    --split val

# Use OpenAI instead of Claude
python scripts/batch_analyze_roco.py \
    --api-choice openai \
    --num-samples 5

# Custom output location
python scripts/batch_analyze_roco.py \
    --output ./my_analysis/results.csv \
    --num-samples 15

# Use training set
python scripts/batch_analyze_roco.py \
    --split train \
    --num-samples 100
```

### Results Format

**CSV columns:**
- `image_id`: Image identifier
- `modality`: Imaging modality
- `ground_truth_caption`: Original caption from medical literature
- `ai_analysis`: AI-generated radiology report
- `latency`: Processing time (seconds)
- `input_tokens`: API input tokens
- `output_tokens`: API output tokens
- `split`: Dataset split used
- `api_provider`: API used (anthropic/openai)
- `model`: Specific model name
- `timestamp`: When analysis was performed

## 💰 Cost Estimation

Using Claude 3.5 Haiku (recommended):

- **Per image analysis**: ~$0.003 - $0.005
- **10 images**: ~$0.03 - $0.05
- **100 images**: ~$0.30 - $0.50
- **1000 images**: ~$3.00 - $5.00

Using Claude Sonnet 4.5 (more accurate):

- **Per image analysis**: ~$0.015 - $0.025
- **10 images**: ~$0.15 - $0.25

## 🔬 Use Cases

### 1. **Model Evaluation**
Compare AI-generated reports with ground truth captions to evaluate model performance.

### 2. **Training Data Generation**
Generate synthetic radiology reports for training downstream models.

### 3. **Research & Development**
Test new prompts, models, or analysis strategies on real medical images.

### 4. **Education**
Learn medical image interpretation by comparing AI analysis with expert captions.

### 5. **Prototype Development**
Build and test radiology AI applications with real-world data.

## 🚨 Important Notes

### ⚠️ Not for Clinical Use

- ROCO is for **research and educational purposes only**
- Do NOT use AI-generated reports for actual medical diagnosis
- Always consult qualified medical professionals for clinical decisions

### 📄 Dataset License

- Check the official Kaggle page for current license terms
- Images sourced from medical literature
- Respect copyright and usage restrictions

### 🔒 Privacy

- ROCO images are de-identified
- No patient-identifiable information included
- Safe for research without HIPAA concerns

## 🐛 Troubleshooting

### Dataset Not Found

```
FileNotFoundError: ROCO dataset not found at ./data/roco/
```

**Solution:**
1. Verify dataset downloaded: `ls ./data/roco/`
2. Check structure includes `images/` folder and CSV files
3. Re-download using `python scripts/download_roco.py`

### Kaggle API Credentials

```
OSError: Could not find kaggle.json
```

**Solution:**
1. Create Kaggle account
2. Go to Account → API → Create New Token
3. Move `kaggle.json` to:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### API Key Issues

```
ValueError: ANTHROPIC_API_KEY not found
```

**Solution:**
1. Check `.env` file has `ANTHROPIC_API_KEY=your_key`
2. Restart Jupyter kernel after updating `.env`
3. Verify API key is valid

## 📚 Additional Resources

- **ROCO Dataset**: https://www.kaggle.com/datasets/shareef0612/roco-radiology
- **Kaggle API Setup**: https://github.com/Kaggle/kaggle-api#api-credentials
- **Claude API Docs**: https://docs.anthropic.com
- **Project Documentation**: [README.md](../README.md)

## 🤝 Contributing

Have improvements for ROCO integration? Submit a PR or open an issue!

---

**Happy Analyzing! 🏥🔬**
