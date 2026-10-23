"""
DrBinati MedAssist - Command Line Interface

Process a complete multi-modal patient case (ECG, medical image, clinical
notes) through the full analysis + RAG pipeline from the command line.

Reconstructed after filesystem corruption zero-filled the original file.
CLI shape and example invocation sourced from PROJECT_SUMMARY.md; call
signatures verified against the intact src/multimodal/analyzers/*.py modules.
"""
import argparse
import base64
import json
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.database.database import db_manager, init_database
from src.multimodal.analyzers.multimodal_analyzer import MultiModalAnalyzer
from src.multimodal.analyzers.rag_system import MultiModalRAG, MultiModalOrchestrator
from src.multimodal.processors.ecg_processor import ECGProcessor
from src.multimodal.processors.clinical_processor import ClinicalNotesProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ecg(ecg_path: Path) -> dict:
    """Load and process an ECG file into multi-modal format"""
    processor = ECGProcessor()
    ecg_signal, _ = processor.load_ecg_file(ecg_path)
    processed = processor.process_ecg(ecg_signal)
    processed['file_path'] = str(ecg_path)
    return processed


def load_image(image_path: Path, modality: str = 'XR') -> dict:
    """Load a medical image into multi-modal format"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    buffered = BytesIO()
    img.save(buffered, format='JPEG')
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        'file_path': str(image_path),
        'modality': modality,
        'image_base64': image_base64,
        'metadata': {'size': img.size, 'mode': img.mode},
        'metadata_text': f"Modality: {modality}\nSize: {img.size}\nMode: {img.mode}"
    }


def load_notes(notes_path: Path) -> dict:
    """Load and process clinical notes into multi-modal format"""
    raw_text = notes_path.read_text(encoding='utf-8')
    processor = ClinicalNotesProcessor(use_presidio=False)
    processed = processor.process_notes(raw_text)
    processed['file_path'] = str(notes_path)
    return processed


def main():
    parser = argparse.ArgumentParser(
        description='DrBinati MedAssist - Multi-modal medical case analysis'
    )
    parser.add_argument('--patient-id', required=True, help='Anonymous patient identifier')
    parser.add_argument('--ecg', type=str, help='Path to ECG file (.dat, .csv, .npy)')
    parser.add_argument('--image', type=str, help='Path to medical image file')
    parser.add_argument('--image-modality', type=str, default='XR', help='Imaging modality (XR, CT, MR, US)')
    parser.add_argument('--notes', type=str, help='Path to clinical notes text file')
    parser.add_argument('--api', type=str, default='anthropic', choices=['anthropic', 'openai'], help='AI provider to use')
    parser.add_argument('--condition', type=str, default=None, help='Optional condition focus (e.g. acute_mi, arrhythmia)')
    parser.add_argument('--no-rag', action='store_true', help='Disable retrieval-augmented context')
    parser.add_argument('--output', type=str, default='./output', help='Directory to write results to')

    args = parser.parse_args()

    if not any([args.ecg, args.image, args.notes]):
        parser.error('At least one of --ecg, --image, --notes must be provided')

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure database tables exist
    init_database()

    logger.info(f"Processing case for patient {args.patient_id}")

    ecg_data = load_ecg(Path(args.ecg)) if args.ecg else None
    imaging_data = load_image(Path(args.image), args.image_modality) if args.image else None
    notes_data = load_notes(Path(args.notes)) if args.notes else None

    analyzer = MultiModalAnalyzer(api_choice=args.api)
    rag = MultiModalRAG()
    orchestrator = MultiModalOrchestrator(analyzer=analyzer, rag=rag)

    with db_manager.get_session() as session:
        results = orchestrator.analyze_with_context(
            session=session,
            patient_id=args.patient_id,
            ecg_data=ecg_data,
            imaging_data=imaging_data,
            notes_data=notes_data,
            use_similar_cases=not args.no_rag,
            condition_focus=args.condition
        )

    results_file = output_dir / f"{args.patient_id}_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Analysis complete. Results written to {results_file}")

    cost_summary = results.get('cost_summary', {})
    print("\n=== Analysis Summary ===")
    print(f"Patient ID: {args.patient_id}")
    print(f"Modalities analyzed: {list(results['analyses'].keys())}")
    print(f"Similar cases retrieved: {len(results.get('similar_cases', []))}")
    print(f"Total cost: ${cost_summary.get('total_cost', 0):.4f}")
    print(f"\nSynthesis:\n{results['synthesis']['synthesis']}")


if __name__ == '__main__':
    main()
