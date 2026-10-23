"""
RAG (Retrieval-Augmented Generation) System for Multi-Modal Healthcare
Uses PostgreSQL with pgvector for similarity search
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func, text
from sentence_transformers import SentenceTransformer
import logging

from src.database.models import (
    Patient, MedicalRecord, ECGRecord, ImagingRecord,
    ClinicalNote, MultiModalAnalysis, ModalityType, SimilarCaseCache
)
from src.database.database import db_manager

logger = logging.getLogger(__name__)


class MedicalEmbeddingModel:
    """Generate embeddings for medical data"""

    def __init__(self, model_name: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model for biomedical text
                       Options:
                       - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" (768 dim)
                       - "sentence-transformers/all-MiniLM-L6-v2" (384 dim)
                       - "BAAI/bge-small-en-v1.5" (384 dim)
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise


class MultiModalRAG:
    """Retrieval-Augmented Generation for healthcare data"""

    def __init__(self, embedding_model: Optional[MedicalEmbeddingModel] = None):
        """
        Initialize RAG system
        
        Args:
            embedding_model: Pre-initialized embedding model. If None, creates default.
        """
        self.embedding_model = embedding_model or MedicalEmbeddingModel()
        logger.info("Initialized MultiModalRAG system")

    def store_medical_record(
        self,
        session: Session,
        patient_id: str,
        modality: ModalityType,
        file_path: str,
        metadata: Dict,
        generate_embedding: bool = True
    ) -> MedicalRecord:
        """
        Store medical record with embedding
        
        Args:
            session: Database session
            patient_id: Patient ID
            modality: Type of medical data
            file_path: Path to file
            metadata: Metadata dict
            generate_embedding: Whether to generate embedding
        
        Returns:
            Created MedicalRecord
        """
        try:
            # Create text representation for embedding
            text_repr = self._create_text_representation(modality, metadata)

            # Generate embedding
            embedding = None
            if generate_embedding and text_repr:
                embedding = self.embedding_model.embed_text(text_repr)
                embedding = embedding.tolist()

            # Create database record
            record = MedicalRecord(
                id=self._generate_id(),
                patient_id=patient_id,
                modality=modality,
                file_path=file_path,
                metadata=metadata,
                embedding=embedding
            )

            session.add(record)
            session.commit()

            logger.info(f"Stored medical record: {modality} for patient {patient_id}")
            return record

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store medical record: {e}")
            raise

    def store_analysis(
        self,
        session: Session,
        patient_id: str,
        analysis_data: Dict,
        generate_embedding: bool = True
    ) -> MultiModalAnalysis:
        """
        Store multi-modal analysis results
        
        Args:
            session: Database session
            patient_id: Patient ID
            analysis_data: Analysis results dict
            generate_embedding: Whether to generate embedding for retrieval
        
        Returns:
            Created MultiModalAnalysis
        """
        try:
            analysis = MultiModalAnalysis(
                id=self._generate_id(),
                patient_id=patient_id,
                **analysis_data
            )

            session.add(analysis)
            session.commit()

            logger.info(f"Stored analysis for patient {patient_id}")
            return analysis

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store analysis: {e}")
            raise

    def retrieve_similar_cases(
        self,
        session: Session,
        query_text: str,
        modality_filter: Optional[ModalityType] = None,
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[Tuple[MedicalRecord, float]]:
        """
        Retrieve similar medical cases using vector similarity
        
        Args:
            session: Database session
            query_text: Query description
            modality_filter: Optional filter by modality
            top_k: Number of results to return
            use_cache: Whether to use cached results
        
        Returns:
            List of (MedicalRecord, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_text(query_text)

            # Check cache first
            if use_cache:
                cached = self._check_cache(session, query_embedding, modality_filter, top_k)
                if cached:
                    logger.info("Retrieved similar cases from cache")
                    return cached

            # Vector similarity search using pgvector cosine distance
            query = select(
                MedicalRecord,
                MedicalRecord.embedding.cosine_distance(query_embedding).label('distance')
            )

            # Apply modality filter
            if modality_filter:
                query = query.where(MedicalRecord.modality == modality_filter)

            # Order by similarity and limit
            query = query.order_by('distance').limit(top_k)

            # Execute query
            results = session.execute(query).all()

            # Convert distance to similarity score
            similar_cases = [(record, 1 - distance) for record, distance in results]

            # Cache results for future queries
            if use_cache:
                self._cache_results(session, query_embedding, modality_filter, similar_cases)

            logger.info(f"Retrieved {len(similar_cases)} similar cases")
            return similar_cases

        except Exception as e:
            logger.error(f"Similar case retrieval failed: {e}")
            raise

    def retrieve_patient_history(
        self,
        session: Session,
        patient_id: str,
        modality_filter: Optional[ModalityType] = None
    ) -> List[MedicalRecord]:
        """
        Retrieve all records for a patient
        
        Args:
            session: Database session
            patient_id: Patient ID
            modality_filter: Optional filter by modality
        
        Returns:
            List of MedicalRecord
        """
        try:
            query = select(MedicalRecord).where(MedicalRecord.patient_id == patient_id)

            if modality_filter:
                query = query.where(MedicalRecord.modality == modality_filter)

            query = query.order_by(MedicalRecord.created_at.desc())

            results = session.execute(query).scalars().all()

            logger.info(f"Retrieved {len(results)} records for patient {patient_id}")
            return results

        except Exception as e:
            logger.error(f"Patient history retrieval failed: {e}")
            raise

    def retrieve_analyses(
        self,
        session: Session,
        patient_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MultiModalAnalysis]:
        """
        Retrieve past analyses
        
        Args:
            session: Database session
            patient_id: Optional patient ID filter
            limit: Max number of results
        
        Returns:
            List of MultiModalAnalysis
        """
        try:
            query = select(MultiModalAnalysis)

            if patient_id:
                query = query.where(MultiModalAnalysis.patient_id == patient_id)

            query = query.order_by(MultiModalAnalysis.created_at.desc()).limit(limit)

            results = session.execute(query).scalars().all()

            logger.info(f"Retrieved {len(results)} analyses")
            return results

        except Exception as e:
            logger.error(f"Analysis retrieval failed: {e}")
            raise

    def search_by_diagnosis(
        self,
        session: Session,
        diagnosis: str,
        top_k: int = 10
    ) -> List[MultiModalAnalysis]:
        """
        Search for cases with similar diagnoses
        
        Args:
            session: Database session
            diagnosis: Diagnosis to search for
            top_k: Number of results
        
        Returns:
            List of MultiModalAnalysis
        """
        try:
            # Case-insensitive partial match on primary diagnosis
            query = select(MultiModalAnalysis).where(
                MultiModalAnalysis.primary_diagnosis.ilike(f"%{diagnosis}%")
            ).order_by(
                MultiModalAnalysis.confidence_score.desc()
            ).limit(top_k)

            results = session.execute(query).scalars().all()

            logger.info(f"Found {len(results)} cases with diagnosis matching '{diagnosis}'")
            return results

        except Exception as e:
            logger.error(f"Diagnosis search failed: {e}")
            raise

    def get_statistics(self, session: Session) -> Dict:
        """Get database statistics"""
        try:
            stats = {
                'total_patients': session.query(func.count(Patient.id)).scalar(),
                'total_records': session.query(func.count(MedicalRecord.id)).scalar(),
                'total_analyses': session.query(func.count(MultiModalAnalysis.id)).scalar(),
                'records_by_modality': {},
                'analyses_by_risk': {}
            }

            # Count records by modality
            for modality in ModalityType:
                count = session.query(func.count(MedicalRecord.id)).filter(
                    MedicalRecord.modality == modality
                ).scalar()
                stats['records_by_modality'][modality.value] = count

            # Count analyses by risk stratification
            for risk in ['low', 'moderate', 'high', 'critical']:
                count = session.query(func.count(MultiModalAnalysis.id)).filter(
                    MultiModalAnalysis.risk_stratification == risk
                ).scalar()
                stats['analyses_by_risk'][risk] = count or 0

            return stats

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            raise

    def _create_text_representation(self, modality: ModalityType, metadata: Dict) -> str:
        """Create text representation for embedding"""
        parts = [f"Medical data type: {modality.value}"]

        # Add modality-specific details
        if modality == ModalityType.ECG:
            if 'heart_rate' in metadata:
                parts.append(f"Heart rate: {metadata['heart_rate']} bpm")
            if 'rhythm' in metadata:
                parts.append(f"Rhythm: {metadata['rhythm']}")

        elif modality in (ModalityType.XRAY, ModalityType.CT, ModalityType.MRI):
            if 'body_part' in metadata:
                parts.append(f"Body part: {metadata['body_part']}")
            if 'modality' in metadata:
                parts.append(f"Imaging modality: {metadata['modality']}")
            if 'study_description' in metadata:
                parts.append(f"Study: {metadata['study_description']}")

        elif modality == ModalityType.CLINICAL_NOTES:
            if 'chief_complaint' in metadata:
                parts.append(f"Chief complaint: {metadata['chief_complaint']}")
            if 'assessment' in metadata:
                parts.append(f"Assessment: {metadata['assessment']}")

        return " | ".join(parts)

    def _check_cache(
        self,
        session: Session,
        query_embedding: np.ndarray,
        modality: Optional[ModalityType],
        top_k: int
    ) -> Optional[List[Tuple[MedicalRecord, float]]]:
        """Check if similar query is cached"""
        try:
            # Cache lookup requires approximate embedding matching
            # Disabled for now - always compute fresh results
            return None
        except:
            return None

    def _cache_results(
        self,
        session: Session,
        query_embedding: np.ndarray,
        modality: Optional[ModalityType],
        results: List[Tuple[MedicalRecord, float]]
    ):
        """Cache query results"""
        try:
            # Cache writes require approximate embedding matching
            # Disabled for now - no-op
            return None
        except:
            return None

    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())


class MultiModalOrchestrator:
    """Orchestrate complete multi-modal analysis with RAG"""

    def __init__(
        self,
        analyzer,
        rag: Optional[MultiModalRAG] = None
    ):
        """
        Initialize orchestrator
        
        Args:
            analyzer: MultiModalAnalyzer instance
            rag: MultiModalRAG instance. If None, creates default.
        """
        self.analyzer = analyzer
        self.rag = rag or MultiModalRAG()
        logger.info("Initialized MultiModalOrchestrator")

    def analyze_with_context(
        self,
        session: Session,
        patient_id: str,
        ecg_data: Optional[Dict] = None,
        imaging_data: Optional[Dict] = None,
        notes_data: Optional[Dict] = None,
        use_similar_cases: bool = True,
        condition_focus: Optional[str] = None
    ) -> Dict:
        """
        Complete analysis with RAG context retrieval
        
        Args:
            session: Database session
            patient_id: Patient ID
            ecg_data: Processed ECG data
            imaging_data: Processed imaging data
            notes_data: Processed clinical notes
            use_similar_cases: Whether to retrieve similar cases for context
            condition_focus: Optional condition focus
        
        Returns:
            Complete analysis results with RAG context
        """
        try:
            results = {
                'patient_id': patient_id,
                'timestamp': None,
                'analyses': {},
                'similar_cases': [],
                'synthesis': None,
                'cost_summary': None
            }

            # Retrieve patient history for context
            patient_history = self.rag.retrieve_patient_history(session, patient_id)

            # Analyze ECG
            if ecg_data:
                ecg_analysis = self.analyzer.analyze_ecg(ecg_data, condition_focus)
                results['analyses']['ecg'] = ecg_analysis

                # Store record
                self.rag.store_medical_record(
                    session=session,
                    patient_id=patient_id,
                    modality=ModalityType.ECG,
                    file_path=ecg_data.get('file_path', ''),
                    metadata=ecg_data.get('metadata', {})
                )

            if imaging_data:
                imaging_analysis = self.analyzer.analyze_medical_image(
                    imaging_data,
                    modality=imaging_data.get('modality', 'XR'),
                    focus=condition_focus
                )
                results['analyses']['imaging'] = imaging_analysis

                # Map imaging modality string to enum
                modality_map = {
                    'XR': ModalityType.XRAY,
                    'CT': ModalityType.CT,
                    'MR': ModalityType.MRI
                }
                self.rag.store_medical_record(
                    session=session,
                    patient_id=patient_id,
                    modality=modality_map.get(imaging_data.get('modality', 'XR'), ModalityType.XRAY),
                    file_path=imaging_data.get('file_path', ''),
                    metadata=imaging_data.get('metadata', {})
                )

            if notes_data:
                clinical_analysis = self.analyzer.analyze_clinical_notes(notes_data)
                results['analyses']['clinical'] = clinical_analysis

                # Store record
                self.rag.store_medical_record(
                    session=session,
                    patient_id=patient_id,
                    modality=ModalityType.CLINICAL_NOTES,
                    file_path=notes_data.get('file_path', ''),
                    metadata=notes_data.get('structured', {})
                )

            # Retrieve similar cases for context
            if use_similar_cases and any([ecg_data, imaging_data, notes_data]):
                # Build query text from available modalities
                query_parts = []
                if ecg_data:
                    query_parts.append(ecg_data.get('annotations', ''))
                if notes_data:
                    query_parts.append(notes_data.get('summary', ''))

                query = " ".join(query_parts)

                similar_cases = self.rag.retrieve_similar_cases(
                    session=session,
                    query_text=query,
                    top_k=3
                )
                results['similar_cases'] = similar_cases

            # Synthesize findings across modalities
            synthesis = self.analyzer.synthesize_multimodal(
                ecg_analysis=results['analyses'].get('ecg', {}).get('analysis', 'N/A'),
                imaging_analysis=results['analyses'].get('imaging', {}).get('analysis', 'N/A'),
                clinical_analysis=results['analyses'].get('clinical', {}).get('analysis', 'N/A'),
                context='general'
            )
            results['synthesis'] = synthesis

            # Persist the complete analysis
            analysis_data = {
                'status': 'completed',
                'api_provider': self.analyzer.api_choice,
                'model_used': self.analyzer.model,
                'ecg_analysis': results['analyses'].get('ecg', {}).get('analysis'),
                'imaging_analysis': results['analyses'].get('imaging', {}).get('analysis'),
                'clinical_analysis': results['analyses'].get('clinical', {}).get('analysis'),
                'comprehensive_assessment': synthesis['synthesis'],
                'api_cost': self.analyzer.cost_tracker.get_session_summary()['total_cost']
            }

            self.rag.store_analysis(session, patient_id, analysis_data)

            # Attach cost summary
            results['cost_summary'] = self.analyzer.get_cost_summary()

            return results

        except Exception as e:
            logger.error(f"Orchestrated analysis failed: {e}")
            raise
