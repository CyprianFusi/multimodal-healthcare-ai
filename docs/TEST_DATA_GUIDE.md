# Test Data Guide

Technical reference for the test data suite in `data/test_samples/` — file formats,
byte layouts, metadata schemas, how to load each modality programmatically, and how
to extend the generator.

For a quick "what's in the box" summary, see
[TEST_DATA_READY.md](../TEST_DATA_READY.md).

---

## Contents

25 files, ~2.2 MB total:

| Modality | Directory | Files | Format |
|----------|-----------|-------|--------|
| ECG | `ecg/` | 5 signals + 5 metadata | raw int16 `.dat` + `.json` |
| Clinical notes | `clinical_notes/` | 5 | UTF-8 `.txt` |
| Medical imaging | `medical_imaging/` | 5 images + 5 metadata | RGB JPEG + `.json` |

---

## ECG Test Data

### Binary Layout

Each `.dat` file is a headerless dump of a NumPy `int16` array — there is no WFDB
header, so you must supply the shape yourself.

| Property | Value |
|----------|-------|
| Data type | `int16` (little-endian) |
| Leads | 12 |
| Sampling rate | 500 Hz |
| Duration | 30 s |
| Samples per lead | 15,000 |
| Array shape | `(12, 15000)` — lead-major |
| File size | 360,000 bytes exactly |
| Units | microvolts (µV) |
| Typical range | −2000 to +2000 µV; peak QRS up to ±1500 µV |

**Lead order:** I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

The generator applies a per-lead amplitude scaling of `1.0 + (lead_index - 6) * 0.1`
so the leads are not identical copies, and for the ST-elevation case adds a +0.2 mV
offset to leads at indices 2–5 (III, aVR, aVL, aVF in the array ordering above).

### Loading

```python
import numpy as np

ecg = np.fromfile(
    'data/test_samples/ecg/ecg_sample_1_normal.dat',
    dtype=np.int16
).reshape(12, 15000)

lead_ii = ecg[1] / 1000.0        # back to millivolts
```

Through the project's own processor:

```python
from src.multimodal.processors.ecg_processor import ECGProcessor

processor = ECGProcessor()
signal, meta = processor.load_ecg_file('data/test_samples/ecg/ecg_sample_1_normal.dat')
processed = processor.process_ecg(signal)
```

### Metadata Schema

```json
{
  "filename": "ecg_sample_1_normal.dat",
  "pattern": "normal",
  "description": "Normal sinus rhythm",
  "sampling_rate": 500,
  "duration": 30,
  "num_leads": 12,
  "heart_rate": 75,
  "generated": "2026-01-01T10:00:33.761405"
}
```

### Patterns

| Sample | Pattern | HR | Noise | Description |
|--------|---------|-----|-------|-------------|
| 1 | `normal` | 75 | 0.01 | Normal sinus rhythm |
| 2 | `tachycardia` | 120 | 0.02 | Sinus tachycardia |
| 3 | `bradycardia` | 45 | 0.01 | Sinus bradycardia |
| 4 | `irregular` | 85 | 0.03 | Irregular rhythm, possible AF |
| 5 | `st_elevation` | 95 | 0.01 | ST elevation, possible MI |

Signals come from `neurokit2.ecg_simulate` with `random_state=42 + i*12 + lead_idx`,
so regeneration is deterministic.

### Expected Analysis Output

For `ecg_sample_1_normal.dat` a correct analysis reports normal sinus rhythm at
~75 bpm, PR ~160 ms, QRS ~90 ms, QTc ~420 ms, isoelectric ST segments, upright T
waves in I/II/V3–V6, and no acute abnormality.

For `ecg_sample_5_st_elevation.dat` it should report sinus rhythm at ~95 bpm with
2–3 mm ST elevation in the anterior leads, findings consistent with acute anterior
STEMI, and recommend urgent cardiology involvement.

---

## Clinical Notes Test Data

### Format

Plain UTF-8 text, 1.1–1.5 KB each, ~380–450 words. Every note carries the same
uppercase section headers, which is what the section extractor keys off:

```
CHIEF COMPLAINT:
HISTORY OF PRESENT ILLNESS:
PAST MEDICAL HISTORY:
MEDICATIONS:
PHYSICAL EXAMINATION:
VITAL SIGNS:
ASSESSMENT:
PLAN:
```

### Loading

```python
from src.multimodal.processors.clinical_processor import ClinicalNotesProcessor

raw = open('data/test_samples/clinical_notes/clinical_note_1_acute_mi_case.txt',
           encoding='utf-8').read()

processor = ClinicalNotesProcessor(use_presidio=False)
processed = processor.process_notes(raw)
```

Set `use_presidio=True` to exercise the PHI de-identification path. The test notes
contain no real PHI, but they do contain names of medications, ages, and vital signs
that make useful de-identification smoke tests.

### Cases

| # | Condition | Diagnostic anchors |
|---|-----------|-------------------|
| 1 | Acute MI | 2 h substernal pressure, radiation to left arm, diaphoresis, HTN + hyperlipidaemia, aspirin + heparin given |
| 2 | Pneumonia | 3 d fever/productive cough, RLL crackles, 38.9 °C, SpO2 93% RA |
| 3 | Stroke | Sudden left hemiparesis, facial droop, dysarthria, NIHSS 12, 45 min from onset |
| 4 | CHF exacerbation | Known EF 25%, orthopnoea, JVD, S3, 2+ pitting oedema, +4 kg |
| 5 | DKA | Glucose 485 mg/dL, pH 7.12, ketones 4+, Kussmaul respirations |

---

## Medical Imaging Test Data

### Format

RGB JPEG, sampled from the ROCO **validation** split with `random_state=42`.
Dimensions vary by source figure (roughly 27 KB–124 KB per file).

### Metadata Schema

```json
{
  "filename": "roco_sample_1.jpg",
  "roco_id": "ROCO_11519",
  "original_filename": "PMC3445094_10.1177_1941738110368392-fig16.jpg",
  "caption": "Anteroposterior radiograph of a right hip. A herniation pit is present (arrow), often associated with cam impingement.",
  "keywords": "hip,right,impingement,herniation,radiograph,cam,anteroposterior,pit,present",
  "extracted": "2025-12-31T20:48:07.161118",
  "source": "ROCO Radiology Dataset"
}
```

The `caption` field is the ground truth written by the authors of the source
publication — use it to score AI output rather than as an input to the model.

### Loading

```python
import json
from PIL import Image

img = Image.open('data/test_samples/medical_imaging/roco_sample_1.jpg')
meta = json.load(open('data/test_samples/medical_imaging/roco_sample_1.json'))

ground_truth = meta['caption']
```

Regenerating images requires the ROCO dataset at `./data/roco/` (see
[../DOWNLOAD_GUIDE.md](../DOWNLOAD_GUIDE.md)). Without it, `--imaging` prints a
warning and skips rather than failing — the five shipped images remain in place.

---

## Multi-Modal Test Scenarios

### Concordant: Acute MI

```bash
python main.py \
  --patient-id TEST_MI_001 \
  --ecg data/test_samples/ecg/ecg_sample_5_st_elevation.dat \
  --notes data/test_samples/clinical_notes/clinical_note_1_acute_mi_case.txt \
  --api anthropic --condition acute_mi --output ./output
```

Both modalities point at STEMI; a correct synthesis states that the ECG corroborates
the clinical presentation and escalates urgency.

### Discordant (negative control)

Pair `ecg_sample_1_normal.dat` with `clinical_note_1_acute_mi_case.txt`. A good
synthesis flags the mismatch rather than smoothing over it — useful for checking the
model is not simply agreeing with whichever modality it saw last.

### RAG behaviour

Run the same patient case twice with and without `--no-rag`. The second run should
retrieve the first as a similar case; `similar_cases` in the output JSON shows what
was pulled in.

---

## Regeneration and Customisation

```bash
python scripts/generate_test_data.py --all --num-samples 5
python scripts/generate_test_data.py --ecg --num-samples 10
python scripts/generate_test_data.py --all --output-dir ./custom_tests
```

Flags: `--all`, `--ecg`, `--imaging`, `--clinical`, `--num-samples` (default 5),
`--output-dir` (default `./data/test_samples`). Passing no modality flag generates
all three.

**Adding an ECG pattern** — add an entry to `ECG_PATTERNS` in
`scripts/generate_test_data.py` with `hr`, `noise`, and `description`. Patterns are
cycled by index, so `--num-samples` beyond the pattern count repeats them.

**Adding a clinical case** — append a dict with `filename` and `text` to
`CLINICAL_NOTE_TEMPLATES`. Keep the uppercase section headers so section extraction
keeps working.

**Changing image sampling** — edit `generate_roco_samples()` to filter the
validation CSV by `Keywords` before sampling, or point it at a different split.

---

## Known Limitations

- **ECG**: simplified pathology, no motion or mains artefact, no arrhythmia
  complexity beyond rate/rhythm changes, no inter-patient variation
- **Notes**: template-based, idealised structure, low linguistic variability
- **Images**: only five samples; ground truth is a figure caption, not a structured
  radiology report; modality coverage is whatever the random draw produced
- **Overall**: adequate for pipeline and regression testing; **not** adequate for
  accuracy claims

For validation work, supplement with PhysioNet (MIT-BIH Arrhythmia, PTB Diagnostic),
CheXpert or MIMIC-CXR for imaging, and i2b2/n2c2 for clinical text.

---

**All test data is for research and educational purposes only. It must not be used
for clinical decision-making.**
