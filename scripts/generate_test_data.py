"""
Generate synthetic test data for DrBinati MedAssist: synthetic ECG recordings,
template-based clinical notes, and sampled ROCO medical images.

Reconstructed after filesystem corruption zero-filled the original file.
Sources used for reconstruction: dissertation/appendices/appendix_b_code_samples.md
(B.4.1, verbatim ECG generation logic) and appendix_c_test_data.md (C.2-C.4 test
case specifications, exact word counts and clinical elements).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ECG_PATTERNS = {
    'normal': {
        'hr': 75,
        'noise': 0.01,
        'description': 'Normal sinus rhythm'
    },
    'tachycardia': {
        'hr': 120,
        'noise': 0.02,
        'description': 'Sinus tachycardia'
    },
    'bradycardia': {
        'hr': 45,
        'noise': 0.01,
        'description': 'Sinus bradycardia'
    },
    'irregular': {
        'hr': 85,
        'noise': 0.03,
        'description': 'Irregular rhythm, possible atrial fibrillation'
    },
    'st_elevation': {
        'hr': 95,
        'noise': 0.01,
        'description': 'ST segment elevation, possible MI'
    }
}


def generate_synthetic_ecg(output_dir: Path, num_samples: int = 5):
    """Generate synthetic ECG signal files"""
    import neurokit2 as nk

    ecg_dir = output_dir / 'ecg'
    ecg_dir.mkdir(parents=True, exist_ok=True)

    pattern_names = list(ECG_PATTERNS.keys())

    for i in range(num_samples):
        pattern_name = pattern_names[i % len(pattern_names)]
        pattern = ECG_PATTERNS[pattern_name]

        sampling_rate = 500  # Hz
        duration = 30  # seconds
        hr = pattern['hr']

        leads = []
        for lead_idx in range(12):
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=hr,
                noise=pattern['noise'],
                random_state=42 + i * 12 + lead_idx
            )

            lead_variation = 1.0 + (lead_idx - 6) * 0.1
            ecg = ecg * lead_variation

            if pattern_name == 'st_elevation' and lead_idx in [2, 3, 4, 5]:
                ecg = ecg + 0.2

            leads.append(ecg)

        ecg_data = np.array(leads)
        ecg_scaled = (ecg_data * 1000).astype(np.int16)

        output_file = ecg_dir / f'ecg_sample_{i+1}_{pattern_name}.dat'
        ecg_scaled.tofile(output_file)

        metadata = {
            'filename': output_file.name,
            'pattern': pattern_name,
            'description': pattern['description'],
            'sampling_rate': sampling_rate,
            'duration': duration,
            'num_leads': 12,
            'heart_rate': hr,
            'generated': datetime.now().isoformat()
        }

        metadata_file = ecg_dir / f'ecg_sample_{i+1}_{pattern_name}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Created: {output_file.name} ({pattern_name})")

    print(f"Generated {num_samples} ECG files in {ecg_dir}")
    return ecg_dir


CLINICAL_NOTE_TEMPLATES = [
    {
        'filename': 'clinical_note_1_acute_mi_case.txt',
        'text': """CHIEF COMPLAINT:
Acute onset substernal chest pressure

HISTORY OF PRESENT ILLNESS:
65-year-old male with a history of hypertension and hyperlipidemia presents
with acute onset substernal chest pressure that began approximately 2 hours
ago while at rest. The pain is described as a pressure/tightness sensation,
radiating to the left arm. Associated symptoms include diaphoresis and
shortness of breath. Patient denies any similar prior episodes. No recent
trauma or exertional trigger identified.

PAST MEDICAL HISTORY:
- Hypertension, diagnosed 10 years ago
- Hyperlipidemia, diagnosed 8 years ago
- No prior cardiac events

MEDICATIONS:
- Lisinopril 10mg daily
- Atorvastatin 40mg daily

PHYSICAL EXAMINATION:
General: Anxious-appearing male, diaphoretic, in moderate distress
Cardiovascular: Regular rate and rhythm, no murmurs, rubs, or gallops
Respiratory: Mild tachypnea, lungs clear to auscultation bilaterally
Extremities: No peripheral edema, pulses intact bilaterally

VITAL SIGNS:
BP: 148/92 mmHg | HR: 95 bpm | RR: 20/min | SpO2: 96% on room air | Temp: 37.0C

ASSESSMENT:
Presentation highly suspicious for acute coronary syndrome, likely STEMI given
symptom onset, radiation pattern, and associated symptoms.

PLAN:
IMMEDIATE INTERVENTIONS INITIATED:
- Aspirin 325mg administered
- Heparin bolus and infusion
- Nitroglycerin sublingual
- Cardiology consultation and catheterization laboratory activation
- Serial troponin measurements
- 12-lead ECG obtained
""",
    },
    {
        'filename': 'clinical_note_2_pneumonia_case.txt',
        'text': """CHIEF COMPLAINT:
Fever and cough

HISTORY OF PRESENT ILLNESS:
Patient presents with a 3-day history of fever, productive cough, and
progressive shortness of breath. Cough is productive of yellow-green sputum.
Patient reports associated pleuritic chest pain with deep inspiration and
generalized fatigue. No recent travel or sick contacts reported.

PAST MEDICAL HISTORY:
- No significant past medical history
- No prior respiratory illness

MEDICATIONS:
- None regularly

PHYSICAL EXAMINATION:
General: Ill-appearing, mild respiratory distress
Respiratory: Crackles auscultated in the right lower lobe, dullness to
percussion over the same region, productive cough noted during exam
Cardiovascular: Tachycardic, regular rhythm, no murmurs

VITAL SIGNS:
Temp: 38.9C | HR: 108 bpm | RR: 24/min | SpO2: 93% on room air | BP: 122/78 mmHg

ASSESSMENT:
Community-acquired pneumonia, right lower lobe, with hypoxemia.

PLAN:
- Empiric antibiotic therapy initiated
- Supplemental oxygen therapy to maintain SpO2 >92%
- Chest radiograph obtained
- Blood cultures and sputum culture sent
- Monitor respiratory status; admit for observation
""",
    },
    {
        'filename': 'clinical_note_3_stroke_evaluation.txt',
        'text': """CHIEF COMPLAINT:
Left-sided weakness

HISTORY OF PRESENT ILLNESS:
Patient presents with sudden onset left-sided weakness that began
approximately 45 minutes prior to arrival. Family reports the patient was
last known well 45 minutes ago. Associated symptoms include left facial
droop and slurred speech (dysarthria). No reported loss of consciousness,
seizure activity, or head trauma.

PAST MEDICAL HISTORY:
- Atrial fibrillation, not anticoagulated
- Hypertension

MEDICATIONS:
- Amlodipine 5mg daily

PHYSICAL EXAMINATION:
General: Alert, in mild distress, appears anxious
Neurological: Left facial droop, left arm and leg weakness (3/5 strength),
dysarthria present, NIHSS score 12 (moderate stroke severity)
Cardiovascular: Irregularly irregular rhythm consistent with known atrial
fibrillation

VITAL SIGNS:
BP: 172/96 mmHg | HR: 88 bpm irregular | RR: 18/min | SpO2: 97% | Glucose: 118 mg/dL

ASSESSMENT:
Acute ischemic stroke, left hemiparesis, NIHSS 12, within tPA eligibility
window given onset 45 minutes prior to presentation.

PLAN:
- Acute stroke protocol activated
- STAT non-contrast head CT obtained to rule out hemorrhage
- tPA eligibility assessment in progress
- Neurology consultation
- Continuous cardiac and neurological monitoring
""",
    },
    {
        'filename': 'clinical_note_4_chf_exacerbation.txt',
        'text': """CHIEF COMPLAINT:
Shortness of breath and leg swelling

HISTORY OF PRESENT ILLNESS:
Patient with known congestive heart failure (ejection fraction 25%) presents
with progressive shortness of breath and bilateral leg swelling over the
past week. Reports orthopnea requiring three pillows to sleep and paroxysmal
nocturnal dyspnea. Notes recent dietary indiscretion and inconsistent
medication adherence.

PAST MEDICAL HISTORY:
- Chronic systolic heart failure, EF 25%
- Coronary artery disease

MEDICATIONS:
- Furosemide 40mg daily (reports missed doses)
- Metoprolol succinate 50mg daily
- Lisinopril 20mg daily

PHYSICAL EXAMINATION:
General: Mild respiratory distress, able to speak in full sentences
Cardiovascular: Jugular venous distension present, S3 gallop noted
Respiratory: Bilateral crackles at lung bases
Extremities: 2+ bilateral pitting leg edema

VITAL SIGNS:
BP: 138/84 mmHg | HR: 92 bpm | RR: 22/min | SpO2: 94% on room air | Weight: up 4kg from baseline

ASSESSMENT:
Acute on chronic systolic heart failure exacerbation, likely precipitated by
dietary indiscretion and medication non-adherence.

PLAN:
- IV diuretic therapy initiated
- Strict fluid restriction and daily weights
- Sodium-restricted diet
- Continue guideline-directed medical therapy
- Monitor renal function and electrolytes
""",
    },
    {
        'filename': 'clinical_note_5_diabetic_ketoacidosis.txt',
        'text': """CHIEF COMPLAINT:
Nausea, vomiting, and confusion

HISTORY OF PRESENT ILLNESS:
Patient with type 1 diabetes mellitus presents with a 1-day history of
nausea, vomiting, and progressive confusion. Family reports the patient has
missed several insulin doses over the past 2 days due to illness. Reports
excessive thirst and frequent urination preceding presentation.

PAST MEDICAL HISTORY:
- Type 1 diabetes mellitus, diagnosed in childhood

MEDICATIONS:
- Insulin glargine (reports missed doses)
- Insulin lispro with meals (reports missed doses)

PHYSICAL EXAMINATION:
General: Lethargic, ill-appearing, Kussmaul respirations noted
HEENT: Dry mucous membranes, fruity breath odor
Cardiovascular: Tachycardic, regular rhythm
Neurological: Confused, oriented to person only

VITAL SIGNS:
Glucose: 485 mg/dL | pH: 7.12 | Ketones: 4+ | HR: 118 bpm | RR: 28/min
(Kussmaul) | BP: 102/64 mmHg | Temp: 37.4C

ASSESSMENT:
Diabetic ketoacidosis, severe, precipitated by insulin non-adherence during
intercurrent illness.

PLAN:
- IV insulin drip initiated
- Aggressive IV fluid resuscitation
- Serial electrolyte monitoring, potassium repletion as needed
- Identify and treat precipitating illness
- ICU-level monitoring
""",
    },
]


def generate_clinical_notes(output_dir: Path, num_samples: int = 5):
    """Generate template-based clinical notes for common conditions"""
    notes_dir = output_dir / 'clinical_notes'
    notes_dir.mkdir(parents=True, exist_ok=True)

    for template in CLINICAL_NOTE_TEMPLATES[:num_samples]:
        output_file = notes_dir / template['filename']
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template['text'])
        print(f"  Created: {output_file.name}")

    print(f"Generated {min(num_samples, len(CLINICAL_NOTE_TEMPLATES))} clinical note files in {notes_dir}")
    return notes_dir


def generate_roco_samples(output_dir: Path, num_samples: int = 5, roco_dir: Path = Path('./data/roco')):
    """Sample medical images from the ROCO dataset with metadata"""
    from PIL import Image
    import pandas as pd

    imaging_dir = output_dir / 'medical_imaging'
    imaging_dir.mkdir(parents=True, exist_ok=True)

    val_csv = roco_dir / 'val_data.csv'
    if not val_csv.exists():
        print(f"  ROCO validation CSV not found at {val_csv}, skipping image sampling")
        return imaging_dir

    df = pd.read_csv(val_csv)
    sampled = df.sample(n=min(num_samples, len(df)), random_state=42)

    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        src_image = roco_dir / 'images' / 'val_set' / row['PMC_ID']
        if not src_image.exists():
            continue

        dest_image = imaging_dir / f'roco_sample_{i}.jpg'
        with Image.open(src_image) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(dest_image, format='JPEG')

        metadata = {
            'pmc_id': row['PMC_ID'],
            'caption': row['Caption'],
            'keywords': row['Keywords'],
            'source': 'ROCO dataset',
            'split': 'validation'
        }
        metadata_file = imaging_dir / f'roco_sample_{i}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Created: {dest_image.name}")

    print(f"Generated ROCO sample images in {imaging_dir}")
    return imaging_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic test data for DrBinati MedAssist'
    )
    parser.add_argument('--all', action='store_true', help='Generate all modalities')
    parser.add_argument('--ecg', action='store_true', help='Generate ECG test data')
    parser.add_argument('--imaging', action='store_true', help='Generate medical imaging test data')
    parser.add_argument('--clinical', action='store_true', help='Generate clinical notes test data')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples per modality')
    parser.add_argument('--output-dir', type=str, default='./data/test_samples', help='Output directory')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ecg = args.all or args.ecg
    run_imaging = args.all or args.imaging
    run_clinical = args.all or args.clinical

    if not (run_ecg or run_imaging or run_clinical):
        run_ecg = run_imaging = run_clinical = True

    if run_ecg:
        print("Generating ECG test data...")
        generate_synthetic_ecg(output_dir, args.num_samples)

    if run_clinical:
        print("Generating clinical notes test data...")
        generate_clinical_notes(output_dir, args.num_samples)

    if run_imaging:
        print("Generating medical imaging test data...")
        generate_roco_samples(output_dir, args.num_samples)

    print("Test data generation complete.")


if __name__ == '__main__':
    main()
