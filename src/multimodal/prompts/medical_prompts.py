"""
Medical Condition-Specific Prompts for Multi-Modal Analysis
Specialized prompts for different medical scenarios and conditions
"""
from typing import Dict, Optional
from enum import Enum


class ConditionCategory(str, Enum):
    """Categories of medical conditions"""
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    ONCOLOGICAL = "oncological"
    MUSCULOSKELETAL = "musculoskeletal"
    GASTROINTESTINAL = "gastrointestinal"
    GENERAL = "general"


class PromptLibrary:
    """Library of specialized medical prompts"""

    # ECG Analysis Prompts

    ECG_GENERAL = """You are an expert cardiologist analyzing a 12-lead ECG.

ECG Context:
{annotations}

Metadata: {metadata}

Please analyze this ECG image systematically and provide:

1. **Rhythm Analysis**
   - Identify the underlying rhythm (sinus, atrial fibrillation, atrial flutter, other)
   - Assess rate and regularity
   - Identify any ectopic beats (PACs, PVCs) or escape beats
   - Note any conduction blocks (AV blocks, bundle branch blocks)

2. **Interval Measurements**
   - PR interval: Normal (120-200 ms), prolonged, or shortened?
   - QRS duration: Normal (<120 ms), widened, or narrow?
   - QT/QTc interval: Normal, prolonged (>450 ms men, >460 ms women), or shortened?
   - Comment on clinical significance of abnormalities

3. **Electrical Axis**
   - Determine QRS axis: Normal (-30° to +90°), left axis deviation, right axis deviation, or extreme axis
   - Clinical implications

4. **Chamber Assessment**
   - Evidence of atrial enlargement (P wave morphology)?
   - Evidence of ventricular hypertrophy (voltage criteria)?
   - Specify which chamber(s) if present

5. **ST Segment and T Wave Analysis**
   - ST segment: Elevated, depressed, or normal in each lead?
   - T waves: Upright, inverted, flattened, or peaked?
   - Patterns suggesting ischemia, injury, or infarction?
   - Specify affected territories (anterior, inferior, lateral, etc.)

6. **Clinical Interpretation**
   - Overall ECG interpretation and severity
   - Possible diagnoses based on findings
   - Clinical urgency (normal, abnormal but stable, urgent, emergent)
   - Recommended immediate actions if any

7. **Comparison and Context**
   - Are findings acute or likely chronic?
   - What additional clinical information would be helpful?
   - Suggestions for further diagnostic testing

Be specific, cite ECG leads when making observations, and provide evidence-based interpretations. Always consider clinical context and recommend appropriate follow-up."""

    ECG_ACUTE_MI = """You are an expert cardiologist evaluating an ECG for ACUTE MYOCARDIAL INFARCTION (MI).

ECG Context:
{annotations}

Metadata: {metadata}

This is a PRIORITY analysis for potential acute coronary syndrome. Focus on:

1. **STEMI Criteria Assessment**
   - ST elevation ≥1mm in at least 2 contiguous leads?
   - Which coronary territory: Anterior (V1-V4), Inferior (II, III, aVF), Lateral (I, aVL, V5-V6), Posterior (V7-V9)?
   - New LBBB (Sgarbossa criteria)?

2. **NSTEMI Indicators**
   - ST depression (≥0.5mm)?
   - T wave inversions (≥1mm)?
   - Hyperacute T waves?

3. **High-Risk Features**
   - Reciprocal changes?
   - Right ventricular involvement (V4R)?
   - Electrical alternans?
   - Complete heart block or new bundle branch block?

4. **Time-Sensitive Actions**
   - Is this STEMI requiring immediate cath lab activation?
   - Time window for reperfusion therapy?
   - Contraindications to thrombolysis if PCI not available?

5. **Risk Stratification**
   - Killip classification based on available data
   - Predicted coronary artery involved
   - Extent of myocardium at risk

Provide IMMEDIATE, ACTIONABLE recommendations. Lives depend on accurate interpretation."""

    ECG_ARRHYTHMIA = """You are an expert cardiac electrophysiologist analyzing an ECG for ARRHYTHMIAS.

ECG Context:
{annotations}

Metadata: {metadata}

Perform detailed arrhythmia analysis:

1. **Rhythm Identification**
   - Precise rhythm diagnosis
   - Atrial rhythm vs ventricular rhythm
   - Regular vs irregular pattern
   - Rate classification (brady, normal, tachy)

2. **Arrhythmia Characterization**
   - If atrial fibrillation: rate control status?
   - If supraventricular tachycardia: specific type (AVNRT, AVRT, AT)?
   - If ventricular tachycardia: monomorphic vs polymorphic?
   - If bradycardia: sinus vs heart block?

3. **Stroke Risk (if AF)**
   - CHA2DS2-VASc considerations
   - Need for anticoagulation?

4. **Hemodynamic Stability Indicators**
   - Heart rate appropriate for clinical situation?
   - Signs of hemodynamic compromise?

5. **Management Recommendations**
   - Immediate interventions needed?
   - Rate vs rhythm control strategy?
   - Need for electrophysiology consultation?
   - Consideration for cardioversion or ablation?

Focus on identifying life-threatening arrhythmias and providing evidence-based management guidance."""

    # Medical Imaging Prompts

    IMAGING_CHEST_XRAY = """You are an expert thoracic radiologist analyzing a chest radiograph.

Imaging Context:
{metadata}

Provide a structured radiology report using standard conventions:

1. **Technical Assessment**
   - Projection (PA, AP, lateral)
   - Penetration (adequate, underpenetrated, overpenetrated)
   - Inspiration (adequate if ≥10 posterior ribs visible)
   - Rotation (clavicles equidistant from spinous processes)
   - Any technical limitations

2. **Systematic Review**
   A. **Airways**
      - Tracheal position and caliber
      - Main bronchi patent
   
   B. **Breathing (Lungs)**
      - Lung volumes and expansion
      - Parenchymal opacities (consolidation, ground-glass, nodules, masses)
      - Distribution (lobar, segmental, diffuse)
      - Air bronchograms present?
      
   C. **Circulation (Heart)**
      - Cardiac size (cardiothoracic ratio <0.5 normal)
      - Cardiac contour abnormalities
      - Mediastinal contour and width
      
   D. **Diaphragm**
      - Position and contour
      - Costophrenic angles (sharp or blunted)
      - Pneumoperitoneum?
      
   E. **Everything Else**
      - Pleural spaces (effusions, pneumothorax)
      - Bones (fractures, lytic/blastic lesions)
      - Soft tissues (emphysema, calcifications)
      - Lines and tubes (if present)

3. **Findings Summary**
   - List all abnormal findings with locations
   - Quantify when possible (size, extent)

4. **Impression**
   - Primary diagnosis
   - Differential diagnoses ranked by likelihood
   - Clinical significance of findings

5. **Recommendations**
   - Need for comparison with prior studies
   - Suggested additional imaging (CT, ultrasound)
   - Clinical correlation needed
   - Follow-up imaging timeline if appropriate

Use Fleischner Society guidelines for pulmonary nodules. Note any urgent findings requiring immediate clinical attention."""

    IMAGING_CT_CHEST = """You are an expert thoracic radiologist analyzing a chest CT scan.

Imaging Context:
{metadata}

Provide comprehensive CT chest interpretation:

1. **Technical Parameters**
   - Contrast: Non-contrast, IV contrast, CT angiography?
   - Phase: Arterial, venous, delayed?
   - Reconstruction: Slice thickness, algorithm
   - Image quality assessment

2. **Systematic Analysis**
   A. **Mediastinum**
      - Vascular structures (aorta, pulmonary arteries, veins)
      - Lymph nodes (size, location, pathological if >1cm short axis)
      - Thymus and thyroid
      - Esophagus
      
   B. **Lungs and Airways**
      - Parenchymal patterns (consolidation, ground-glass, nodules, cavitations, interstitial disease)
      - Apply Fleischner Society classification for nodules
      - Airways (bronchiectasis, wall thickening)
      - Emphysema type and distribution
      
   C. **Pleura and Chest Wall**
      - Pleural effusions (size, complexity)
      - Pleural thickening or masses
      - Pneumothorax
      - Chest wall integrity
      - Breast tissue evaluation
      
   D. **Cardiac Assessment**
      - Cardiac size and chambers
      - Pericardial effusion or thickening
      - Coronary artery calcification (if visible)
      
   E. **Upper Abdomen**
      - Liver, spleen, adrenal glands
      - Upper pole kidneys
      - Incidental findings

3. **Key Measurements**
   - Nodule sizes (use lung windows)
   - Lymph node short axis
   - Effusion volume estimation

4. **Impression and Differential**
   - Primary findings with clinical significance
   - Differential diagnoses with supporting/refuting features
   - Lung-RADS or other classification if applicable

5. **Recommendations**
   - Need for biopsy or intervention
   - Follow-up imaging protocol and timeline
   - Comparison with prior imaging
   - Additional sequences or modalities needed

Report urgent/emergent findings first (pneumothorax, PE, aortic dissection, large mass effect)."""

    IMAGING_ONCOLOGY = """You are an expert oncologic radiologist analyzing medical imaging for cancer assessment.

Imaging Context:
{metadata}

Provide oncology-focused imaging interpretation:

1. **Primary Lesion Assessment**
   - Location, size (3 dimensions)
   - Morphology (well-defined vs infiltrative)
   - Density/signal characteristics
   - Enhancement pattern
   - Invasion of adjacent structures

2. **Staging Evaluation** (apply relevant TNM classification)
   - T: Primary tumor extent
   - N: Regional lymph nodes (location, size, number)
   - M: Distant metastases (organs involved)

3. **RECIST 1.1 Measurements** (if applicable)
   - Identify target lesions (up to 5 total, 2 per organ)
   - Measure longest diameter
   - Sum of diameters for baseline/comparison

4. **Response Assessment** (if comparing with baseline)
   - Complete response: Disappearance of all lesions
   - Partial response: ≥30% decrease in sum of diameters
   - Progressive disease: ≥20% increase or new lesions
   - Stable disease: Neither PR nor PD criteria met

5. **Complications and Treatment Effects**
   - Radiation changes
   - Post-surgical changes
   - Treatment-related complications
   - Pseudoprogression vs true progression

6. **Clinical Implications**
   - Resectability assessment
   - Biopsy accessibility and safety
   - Prognosis implications

Provide measurements in mm, note any urgent complications (bleeding, obstruction, perforation), and give clear staging information for treatment planning."""

    # Clinical Notes Prompts

    CLINICAL_NOTES_GENERAL = """You are an experienced internist reviewing comprehensive clinical documentation.

Clinical Documentation:
{notes}

Provide thorough clinical analysis:

1. **Clinical Summary**
   - Patient presentation and chief concern
   - Relevant timeline of symptoms
   - Pertinent medical/surgical/family/social history
   - Current medications and allergies

2. **Physical Examination Findings**
   - Vital signs and their significance
   - Key examination findings (positive and pertinent negatives)
   - Overall clinical appearance

3. **Risk Factor Analysis**
   - Identified cardiovascular risk factors
   - Metabolic risk factors (obesity, diabetes, hypertension, dyslipidemia)
   - Behavioral risks (smoking, alcohol, substance use)
   - Psychosocial factors affecting health

4. **Differential Diagnosis Framework**
   - Most likely diagnosis (with supporting evidence)
   - Important alternative diagnoses to consider
   - Life-threatening diagnoses that must be ruled out
   - Identify diagnostic uncertainty areas

5. **Current Management Assessment**
   - Appropriateness of current diagnostic workup
   - Medication review (indications, interactions, adherence)
   - Preventive care gaps
   - Health maintenance needs

6. **Recommendations**
   - Diagnostic tests still needed
   - Medication optimization suggestions
   - Lifestyle modifications
   - Specialist referrals if indicated
   - Follow-up plan and timeline

7. **Care Coordination**
   - Patient education needs
   - Barriers to care identified
   - Social determinants of health considerations

Extract insights that inform clinical decision-making and identify any concerning patterns or red flags."""

    CLINICAL_NOTES_CARDIOLOGY = """You are a cardiology consultant reviewing clinical notes for cardiovascular assessment.

Clinical Documentation:
{notes}

Provide cardiology-focused analysis:

1. **Cardiovascular History Deep Dive**
   - Symptoms: Chest pain (characterize using OPQRST), dyspnea, palpitations, syncope, edema
   - Functional capacity (NYHA class if heart failure, CCS class if angina)
   - Previous cardiac events and interventions
   - Cardiovascular risk factors (quantify using calculator if possible)

2. **Physical Examination - Cardiovascular Focus**
   - Blood pressure patterns and orthostatic changes
   - Heart sounds (S3, S4, murmurs with grading)
   - Peripheral pulses and circulation
   - Signs of heart failure (JVD, peripheral edema, rales)

3. **Risk Stratification**
   - Calculate ASCVD risk score (if data available)
   - Heart failure risk factors
   - Atrial fibrillation and stroke risk (CHA2DS2-VASc if applicable)
   - Bleeding risk if on anticoagulation (HAS-BLED if applicable)

4. **Medication Management**
   - GDMT (guideline-directed medical therapy) assessment
   - Statin therapy appropriateness
   - Antiplatelet vs anticoagulation decisions
   - Beta-blocker, ACE-I/ARB, mineralocorticoid antagonist indications

5. **Diagnostic Workup Recommendations**
   - ECG indications and findings to look for
   - Stress testing vs coronary imaging
   - Echocardiography indications
   - Cardiac catheterization consideration

6. **Specialized Interventions**
   - Need for EP study or ablation
   - Structural heart disease interventions
   - Advanced heart failure therapies (LVAD, transplant evaluation)
   - Cardiac rehabilitation referral

Focus on guideline-based care (AHA/ACC) and provide evidence-based recommendations with class of recommendation and level of evidence where applicable."""

    # Multi-Modal Synthesis Prompts

    MULTIMODAL_SYNTHESIS = """You are a senior physician synthesizing multi-modal clinical data for comprehensive patient assessment.

=== ECG ANALYSIS ===
{ecg_analysis}

=== IMAGING STUDIES ===
{imaging_analysis}

=== CLINICAL DOCUMENTATION ===
{clinical_analysis}

Provide integrated multi-modal assessment:

1. **Data Concordance Analysis**
   - How do findings across modalities support or contradict each other?
   - Which findings are mutually reinforcing?
   - Identify any significant discrepancies requiring resolution
   - Assess overall coherence of clinical picture

2. **Unified Diagnostic Assessment**
   - Synthesized primary diagnosis based on ALL available data
   - Confidence level: High / Moderate / Low (justify)
   - Supporting evidence from each modality
   - Alternative diagnoses still under consideration
   - Diagnostic gaps requiring additional information

3. **Comprehensive Risk Stratification**
   - Overall patient risk level: Low / Moderate / High / Critical
   - Short-term risks (hours to days)
   - Medium-term risks (weeks to months)
   - Long-term prognostic factors
   - Specific risk scores if applicable

4. **Integrated Management Plan**
   **Immediate Actions** (within hours):
   - Urgent interventions required
   - Emergency department vs inpatient vs outpatient management
   - Consultations needed stat
   
   **Short-term Plan** (days to weeks):
   - Additional diagnostic studies
   - Medication adjustments
   - Monitoring parameters
   
   **Long-term Strategy** (months):
   - Chronic disease management
   - Risk factor modification
   - Preventive care
   - Rehabilitation needs

5. **Clinical Priorities** (Rank top 3-5)
   For each priority:
   - Specific issue
   - Why it's a priority
   - Recommended action
   - Timeline for action
   - Expected outcome

6. **Communication Recommendations**
   - Key points for patient/family discussion
   - Shared decision-making areas
   - Prognostic information to convey
   - Educational needs

7. **Follow-up Strategy**
   - Timeline for reassessment
   - Which modalities to repeat and when
   - Clinical endpoints to monitor
   - Criteria for escalation of care

Synthesize findings into a coherent clinical narrative. Think like an expert clinician considering the whole patient. Identify patterns that might be missed when viewing each data source in isolation. Provide evidence-based, actionable guidance that prioritizes patient safety and optimal outcomes."""

    MULTIMODAL_EMERGENCY = """You are an emergency medicine physician performing RAPID multi-modal assessment for a potentially critically ill patient.

=== ECG ===
{ecg_analysis}

=== IMAGING ===
{imaging_analysis}

=== CLINICAL PRESENTATION ===
{clinical_analysis}

URGENT ASSESSMENT - Time-sensitive analysis:

1. **IMMEDIATE THREAT TO LIFE?**
   - Identify any life-threatening findings
   - Hemodynamic instability indicators
   - Airway/breathing/circulation concerns
   - Neurological emergencies

2. **EMERGENCY DIAGNOSIS**
   - Most likely emergent diagnosis
   - Critical differentials to rule out
   - Findings requiring immediate intervention

3. **ACUITY LEVEL**
   - ESI Level 1 (Resuscitation): Immediate life-threatening
   - ESI Level 2 (Emergent): High risk, severe pain/distress
   - ESI Level 3 (Urgent): Stable but needs multiple resources
   - ESI Level 4-5 (Less urgent/Non-urgent)

4. **IMMEDIATE ACTIONS** (Priority Order)
   a. Resuscitation needs (ABC stabilization)
   b. Immediate medications/interventions
   c. Emergent procedures or consultations
   d. Stat laboratory/imaging needed
   e. Disposition (resuscitation bay, ICU, floor, discharge)

5. **TIME-SENSITIVE TREATMENTS**
   - Door-to-balloon time for STEMI?
   - Thrombolysis window for stroke?
   - Sepsis bundle timing?
   - Other time-critical interventions?

6. **SAFETY NET**
   - What could we be missing?
   - Worse-case scenario preparations
   - Return precautions for patient/family

Focus on immediately actionable information. Lives depend on rapid, accurate assessment."""


    @classmethod
    def get_ecg_prompt(
        cls,
        condition: Optional[str] = None,
        annotations: str = "",
        metadata: Dict = None
    ) -> str:
        """Get specialized ECG analysis prompt"""
        metadata_str = str(metadata) if metadata else "Not provided"

        if condition == "acute_mi":
            return cls.ECG_ACUTE_MI.format(
                annotations=annotations,
                metadata=metadata_str
            )
        elif condition == "arrhythmia":
            return cls.ECG_ARRHYTHMIA.format(
                annotations=annotations,
                metadata=metadata_str
            )
        else:
            return cls.ECG_GENERAL.format(
                annotations=annotations,
                metadata=metadata_str
            )

    @classmethod
    def get_imaging_prompt(
        cls,
        modality: str,
        focus: Optional[str] = None,
        metadata: Dict = None
    ) -> str:
        """Get specialized imaging analysis prompt"""
        metadata_str = str(metadata) if metadata else "Not provided"

        if modality.upper() in ("XR", "CR", "DX") or "xray" in modality.lower():
            return cls.IMAGING_CHEST_XRAY.format(metadata=metadata_str)
        elif modality.upper() == "CT":
            if focus == "oncology":
                return cls.IMAGING_ONCOLOGY.format(metadata=metadata_str)
            else:
                return cls.IMAGING_CT_CHEST.format(metadata=metadata_str)
        elif focus == "oncology":
            return cls.IMAGING_ONCOLOGY.format(metadata=metadata_str)
        else:
            # Default to chest X-ray style structured report
            return cls.IMAGING_CHEST_XRAY.format(metadata=metadata_str)

    @classmethod
    def get_clinical_notes_prompt(
        cls,
        specialty: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """Get specialized clinical notes analysis prompt"""
        if specialty == "cardiology":
            return cls.CLINICAL_NOTES_CARDIOLOGY.format(notes=notes)
        else:
            return cls.CLINICAL_NOTES_GENERAL.format(notes=notes)

    @classmethod
    def get_synthesis_prompt(
        cls,
        context: str = "general",
        ecg_analysis: str = "N/A",
        imaging_analysis: str = "N/A",
        clinical_analysis: str = "N/A"
    ) -> str:
        """Get multimodal synthesis prompt"""
        if context == "emergency":
            return cls.MULTIMODAL_EMERGENCY.format(
                ecg_analysis=ecg_analysis,
                imaging_analysis=imaging_analysis,
                clinical_analysis=clinical_analysis
            )
        else:
            return cls.MULTIMODAL_SYNTHESIS.format(
                ecg_analysis=ecg_analysis,
                imaging_analysis=imaging_analysis,
                clinical_analysis=clinical_analysis
            )
