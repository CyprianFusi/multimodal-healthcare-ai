"""
Clinical Notes Processing Module
De-identifies and structures clinical documentation
"""
import re
from typing import Dict, List, Optional
from pathlib import Path
import logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy

logger = logging.getLogger(__name__)


class ClinicalNotesProcessor:
    """Process and de-identify clinical notes"""

    SECTION_PATTERNS = {
        'chief_complaint': [
            r'(?:Chief Complaint|CC|C/C):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)',
            r'(?:Presenting Complaint):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'history_present_illness': [
            r'(?:History of Present Illness|HPI|H\.P\.I\.):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)',
            r'(?:Present Illness|PI):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'past_medical_history': [
            r'(?:Past Medical History|PMH|P\.M\.H\.|Medical History):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'medications': [
            r'(?:Medications|Meds|Current Medications):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'allergies': [
            r'(?:Allergies|Allergy|Drug Allergies):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'social_history': [
            r'(?:Social History|SH|S\.H\.):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'family_history': [
            r'(?:Family History|FH|F\.H\.):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'review_of_systems': [
            r'(?:Review of Systems|ROS|R\.O\.S\.):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'physical_exam': [
            r'(?:Physical Exam|PE|P\.E\.|Physical Examination|Examination):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'vital_signs': [
            r'(?:Vital Signs|Vitals|VS):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'labs': [
            r'(?:Labs|Laboratory|Lab Results):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'imaging': [
            r'(?:Imaging|Radiology|X-ray|CT|MRI|Ultrasound):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'assessment': [
            r'(?:Assessment|Impression|Diagnosis|Dx):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)',
            r'(?:Assessment and Plan|A&P|A/P):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ],
        'plan': [
            r'(?:Plan|Treatment Plan|P:):\s*(.*?)(?=\n\n|\n[A-Z][a-z]+:|\Z)'
        ]
    }

    def __init__(self, use_presidio: bool = True):
        """
        Initialize processor
        
        Args:
            use_presidio: Whether to use Presidio for de-identification
        """
        self.use_presidio = use_presidio

        if use_presidio:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio de-identification initialized")
            except Exception as e:
                logger.warning(f"Presidio initialization failed: {e}. Using rule-based de-identification.")
                self.use_presidio = False

        # Load SciSpacy model for medical entity recognition
        try:
            self.nlp = spacy.load("en_core_sci_md")
            logger.info("Loaded SciSpacy medical model")
        except Exception as e:
            logger.warning(f"SciSpacy model not available: {e}")
            self.nlp = None

    def process_notes(self, raw_text: str) -> Dict:
        """
        Process clinical notes
        
        Returns:
            Dictionary with:
            - structured: Parsed sections
            - anonymized: De-identified text
            - entities: Extracted medical entities
            - summary: Brief summary
        """
        try:
            # Clean text
            cleaned_text = self._clean_text(raw_text)

            # De-identify
            anonymized_text = self.deidentify(cleaned_text)

            # Extract structured sections
            structured = self.extract_sections(anonymized_text)

            # Extract medical entities
            entities = self._extract_entities(anonymized_text) if self.nlp else {}

            # Generate summary
            summary = self._generate_summary(structured)

            return {
                'structured': structured,
                'anonymized': anonymized_text,
                'entities': entities,
                'summary': summary,
                'metadata': {
                    'sections_found': list(structured.keys()),
                    'total_length': len(anonymized_text),
                    'deidentified': self.use_presidio
                }
            }

        except Exception as e:
            logger.error(f"Notes processing failed: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s\n\.,;:?!()\-/]', '', text)

        return text.strip()

    def deidentify(self, text: str) -> str:
        """
        De-identify PHI from text
        
        Uses Presidio if available, otherwise rule-based approach
        """
        if self.use_presidio:
            return self._deidentify_presidio(text)
        else:
            return self._deidentify_rules(text)

    def _deidentify_presidio(self, text: str) -> str:
        """De-identify using Presidio"""
        try:
            # Analyze text for PHI entities
            results = self.analyzer.analyze(
                text=text,
                entities=[
                    'PERSON',
                    'LOCATION',
                    'DATE_TIME',
                    'PHONE_NUMBER',
                    'EMAIL_ADDRESS',
                    'US_SSN',
                    'MEDICAL_LICENSE',
                    'US_PASSPORT'
                ],
                language='en'
            )

            # Anonymize detected entities
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )

            return anonymized.text

        except Exception as e:
            logger.error(f"Presidio de-identification failed: {e}")
            return self._deidentify_rules(text)

    def _deidentify_rules(self, text: str) -> str:
        """Rule-based de-identification (fallback)"""
        # Dates
        text = re.sub(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            '[DATE]',
            text
        )

        # Phone numbers
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            text
        )

        # Social security numbers
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            text
        )

        # Email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )

        # Ages over 89 must be aggregated under HIPAA Safe Harbor
        def replace_age(match):
            age = int(match.group(1))
            return '[AGE >89]' if age > 89 else match.group(0)

        text = re.sub(
            r'(\d+)(?:\s*years?\s*old|\s*y/?o)',
            replace_age,
            text,
            flags=re.IGNORECASE
        )

        return text

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from clinical notes"""
        sections = {}

        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[section_name] = match.group(1).strip()
                    break

        # Fall back to the raw note when no headers are recognised
        if not sections:
            sections['unstructured'] = text

        return sections

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using SciSpacy"""
        if not self.nlp:
            return {}

        try:
            doc = self.nlp(text)

            entities = {
                'diseases': [],
                'medications': [],
                'procedures': [],
                'anatomy': []
            }

            for ent in doc.ents:
                if ent.label_ in ('DISEASE', 'PROBLEM'):
                    entities['diseases'].append(ent.text)
                elif ent.label_ in ('MEDICATION', 'DRUG'):
                    entities['medications'].append(ent.text)
                elif ent.label_ in ('PROCEDURE', 'TEST'):
                    entities['procedures'].append(ent.text)
                elif ent.label_ in ('ANATOMY', 'ORGAN'):
                    entities['anatomy'].append(ent.text)

            # Deduplicate
            for key in entities:
                entities[key] = list(set(entities[key]))

            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}

    def _generate_summary(self, structured: Dict[str, str]) -> str:
        """Generate brief summary of clinical notes"""
        summary_parts = []

        # Chief complaint
        if 'chief_complaint' in structured:
            cc = structured['chief_complaint'][:200]
            summary_parts.append(f"Chief Complaint: {cc}")

        # Assessment
        if 'assessment' in structured:
            assessment = structured['assessment'][:300]
            summary_parts.append(f"Assessment: {assessment}")

        # Plan
        if 'plan' in structured:
            plan = structured['plan'][:200]
            summary_parts.append(f"Plan: {plan}")

        return '\n\n'.join(summary_parts) if summary_parts else "No summary available"

    def format_for_llm(self, processed_notes: Dict) -> str:
        """Format processed notes for LLM consumption"""
        structured = processed_notes['structured']

        formatted_sections = []

        section_labels = {
            'chief_complaint': 'Chief Complaint',
            'history_present_illness': 'History of Present Illness',
            'past_medical_history': 'Past Medical History',
            'medications': 'Current Medications',
            'allergies': 'Allergies',
            'social_history': 'Social History',
            'family_history': 'Family History',
            'review_of_systems': 'Review of Systems',
            'physical_exam': 'Physical Examination',
            'vital_signs': 'Vital Signs',
            'labs': 'Laboratory Results',
            'imaging': 'Imaging Studies',
            'assessment': 'Assessment/Impression',
            'plan': 'Treatment Plan'
        }

        for key, label in section_labels.items():
            if key in structured and structured[key]:
                formatted_sections.append(f"{label}:\n{structured[key]}\n")

        # Include unparsed documentation verbatim
        if 'unstructured' in structured:
            formatted_sections.append(f"Clinical Documentation:\n{structured['unstructured']}")

        return '\n'.join(formatted_sections)

    def validate_note(self, text: str) -> Dict[str, bool]:
        """Validate clinical note completeness"""
        required_sections = ['chief_complaint', 'history_present_illness', 'assessment']
        structured = self.extract_sections(text)

        validation = {
            'has_required_sections': all(s in structured for s in required_sections),
            'has_chief_complaint': 'chief_complaint' in structured,
            'has_history': 'history_present_illness' in structured,
            'has_assessment': 'assessment' in structured,
            'has_plan': 'plan' in structured,
            'is_deidentified': '[' in text,
            'sufficient_length': len(text) > 100
        }

        return validation


# Utility function for quick processing
def process_clinical_note_file(file_path: Path) -> Dict:
    """Quick function to process a clinical note file"""
    processor = ClinicalNotesProcessor()

    # Read the note from disk
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return processor.process_notes(text)
