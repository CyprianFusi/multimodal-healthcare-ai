# Test Data Samples for Streamlit UI

This directory contains test data for the Multi-Modal Healthcare AI Streamlit interface.

Generated on: 2025-12-31

## 📁 Directory Structure

```
test_samples/
├── ecg/                    # Synthetic ECG signal files
│   ├── ecg_sample_*.dat    # Binary ECG data files
│   └── ecg_sample_*.json   # Metadata for each ECG
│
├── clinical_notes/         # Clinical documentation samples
│   └── clinical_note_*.txt # Text-based clinical notes
│
└── medical_imaging/        # Real radiology images from ROCO
    ├── roco_sample_*.jpg   # Medical images
    └── roco_sample_*.json  # Image metadata with captions
```

## 🧪 ECG Samples (5 files)

Synthetic 12-lead ECG signals representing different cardiac conditions:

1. **ecg_sample_1_normal.dat** - Normal sinus rhythm (HR: 75 bpm)
2. **ecg_sample_2_tachycardia.dat** - Sinus tachycardia (HR: 120 bpm)
3. **ecg_sample_3_bradycardia.dat** - Sinus bradycardia (HR: 45 bpm)
4. **ecg_sample_4_irregular.dat** - Irregular rhythm, possible atrial fibrillation (HR: 85 bpm)
5. **ecg_sample_5_st_elevation.dat** - ST segment elevation, possible MI (HR: 95 bpm)

**Format**: Binary .dat files with 12 leads, 500 Hz sampling rate, 10 seconds duration

**Metadata**: Each .dat file has a corresponding .json file with:
- Pattern description
- Heart rate
- Sampling rate
- Number of leads

## 📝 Clinical Notes (5 files)

Realistic clinical documentation for various medical cases:

1. **clinical_note_1_acute_mi_case.txt** - Acute myocardial infarction (heart attack)
2. **clinical_note_2_pneumonia_case.txt** - Community-acquired pneumonia
3. **clinical_note_3_stroke_evaluation.txt** - Acute ischemic stroke
4. **clinical_note_4_chf_exacerbation.txt** - Congestive heart failure exacerbation
5. **clinical_note_5_diabetic_ketoacidosis.txt** - Diabetic ketoacidosis (DKA)

**Format**: Plain text files with standard clinical note structure:
- Chief Complaint
- History of Present Illness
- Past Medical History
- Medications
- Physical Exam
- Assessment
- Plan

## 🏥 Medical Imaging (5 samples)

Real radiology images extracted from the ROCO dataset:

**Format**: JPEG images at 95% quality

**Metadata**: Each .jpg file has a corresponding .json file with:
- ROCO ID
- Original filename
- Ground truth caption (from medical literature)
- Keywords
- Source information

**Sample images include** various radiology modalities (X-ray, CT, MRI, Ultrasound)

## 🚀 How to Use

### In Streamlit UI

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Upload test files:
   - **ECG**: Upload any `ecg_sample_*.dat` file
   - **Clinical Notes**: Upload any `clinical_note_*.txt` file
   - **Medical Imaging**: Upload any `roco_sample_*.jpg` file

3. The UI will process and analyze the uploaded files using AI

### Regenerate Test Data

To create fresh test samples:

```bash
# Generate all types (5 samples each)
python scripts/generate_test_data.py --all --num-samples 5

# Generate only ECG samples
python scripts/generate_test_data.py --ecg --num-samples 10

# Generate only medical images from ROCO
python scripts/generate_test_data.py --imaging --num-samples 20

# Generate only clinical notes
python scripts/generate_test_data.py --clinical --num-samples 5

# Custom output directory
python scripts/generate_test_data.py --all --output-dir ./my_test_data
```

## 📊 Expected Analysis Results

### ECG Analysis
The AI should identify:
- Heart rate and rhythm
- Presence of arrhythmias
- ST segment changes
- Potential diagnoses (normal, tachycardia, bradycardia, AFib, MI)

### Clinical Notes Analysis
The AI should extract:
- Chief complaint
- Key diagnoses
- Medications
- Vital signs
- Treatment plan
- Critical findings

### Medical Imaging Analysis
The AI should describe:
- Anatomical structures visible
- Any abnormalities present
- Imaging modality used
- Clinical findings
- Comparison with ground truth caption

## 💰 Cost Estimates

Using Claude 3.5 Haiku:
- **ECG analysis**: ~$0.003 per file
- **Clinical notes**: ~$0.004 per file
- **Medical imaging**: ~$0.003 per file

**Total for all 15 samples**: ~$0.05

## ⚠️ Important Notes

### ECG Files
- Synthetic data for testing purposes only
- Not suitable for clinical validation
- Generated patterns are simplified representations

### Clinical Notes
- Fictional patient cases
- Based on realistic clinical scenarios
- No real patient information

### Medical Images
- Real radiology images from ROCO dataset
- Sourced from medical literature
- De-identified, no patient data
- For research and educational use only

## 🔄 Version History

- **v1.0** (2025-12-31): Initial test data generation
  - 5 ECG samples (5 cardiac patterns)
  - 5 clinical notes (5 medical conditions)
  - 5 medical images (from ROCO dataset)

## 📚 References

- **ROCO Dataset**: https://www.kaggle.com/datasets/shareef0612/roco-radiology
- **ECG Simulation**: Based on standard cardiac waveforms
- **Clinical Notes**: Based on standard medical documentation practices

---

**Status**: ✅ Ready for Testing
**Last Updated**: 2025-12-31
**Total Files**: 25 (5 ECG + 5 clinical notes + 5 images + metadata)
