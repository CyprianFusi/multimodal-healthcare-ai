"""
Database models for multi-modal healthcare system
"""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, JSON,
    ForeignKey, Enum, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import enum
from datetime import datetime


Base = declarative_base()


class ModalityType(str, enum.Enum):
    """Supported medical modalities"""
    ECG = "ecg"
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    CLINICAL_NOTES = "clinical_notes"


class AnalysisStatus(str, enum.Enum):
    """Analysis processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Patient(Base):
    """Patient information (anonymized)"""
    __tablename__ = "patients"

    id = Column(String(36), primary_key=True)
    anonymous_id = Column(String(100), unique=True, nullable=False, index=True)
    age = Column(Integer)
    gender = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    medical_records = relationship("MedicalRecord", back_populates="patient")
    analyses = relationship("MultiModalAnalysis", back_populates="patient")


class MedicalRecord(Base):
    """Individual medical records (ECG, imaging, notes)"""
    __tablename__ = "medical_records"

    id = Column(String(36), primary_key=True)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False, index=True)
    modality = Column(Enum(ModalityType), nullable=False, index=True)

    # File storage
    file_path = Column(String(500))
    file_type = Column(String(50))
    file_size = Column(Integer)

    # Processed artifacts
    processed_data_path = Column(String(500))
    thumbnail_path = Column(String(500))

    # Metadata
    record_metadata = Column(JSON)
    acquisition_date = Column(DateTime(timezone=True))

    # Vector embedding for similarity search
    embedding = Column(Vector(1536))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    patient = relationship("Patient", back_populates="medical_records")

    # Vector similarity index
    __table_args__ = (
        Index('idx_medical_records_embedding', 'embedding', postgresql_using='ivfflat'),
    )


class ECGRecord(Base):
    """Specialized ECG signal data"""
    __tablename__ = "ecg_records"

    id = Column(String(36), primary_key=True)
    medical_record_id = Column(String(36), ForeignKey("medical_records.id"), nullable=False)

    # Signal properties
    sampling_rate = Column(Integer)
    duration = Column(Float)
    num_leads = Column(Integer)

    # Extracted features
    heart_rate = Column(Float)
    rr_interval = Column(Float)
    pr_interval = Column(Float)
    qrs_duration = Column(Float)
    qt_interval = Column(Float)
    qtc_corrected = Column(Float)

    # Generated visualizations
    spectrogram_path = Column(String(500))
    waveform_image_path = Column(String(500))

    # Annotations
    annotations = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ImagingRecord(Base):
    """Specialized medical imaging data"""
    __tablename__ = "imaging_records"

    id = Column(String(36), primary_key=True)
    medical_record_id = Column(String(36), ForeignKey("medical_records.id"), nullable=False)

    # Study information
    modality = Column(String(50))
    body_part = Column(String(100))
    study_description = Column(String(500))

    # DICOM metadata
    dicom_metadata = Column(JSON)
    series_number = Column(Integer)
    instance_number = Column(Integer)

    # Image geometry
    pixel_spacing = Column(JSON)
    slice_thickness = Column(Float)
    image_orientation = Column(JSON)

    # Processed images
    normalized_image_path = Column(String(500))
    segmentation_mask_path = Column(String(500))

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ClinicalNote(Base):
    """Clinical documentation"""
    __tablename__ = "clinical_notes"

    id = Column(String(36), primary_key=True)
    medical_record_id = Column(String(36), ForeignKey("medical_records.id"), nullable=False)

    # Structured sections
    chief_complaint = Column(Text)
    history_present_illness = Column(Text)
    past_medical_history = Column(Text)
    medications = Column(Text)
    allergies = Column(Text)
    physical_exam = Column(Text)
    assessment = Column(Text)
    plan = Column(Text)

    # Full text
    raw_text = Column(Text)
    anonymized_text = Column(Text)

    # Provenance
    note_type = Column(String(100))
    author_role = Column(String(100))
    note_date = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MultiModalAnalysis(Base):
    """Complete multi-modal analysis results"""
    __tablename__ = "multimodal_analyses"

    id = Column(String(36), primary_key=True)
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False, index=True)

    # Processing information
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING, index=True)
    api_provider = Column(String(50))
    model_used = Column(String(100))

    # Per-modality analyses
    ecg_analysis = Column(Text)
    imaging_analysis = Column(Text)
    clinical_analysis = Column(Text)

    # Integrated findings
    comprehensive_assessment = Column(Text)
    risk_stratification = Column(String(50))
    primary_diagnosis = Column(String(500))
    differential_diagnoses = Column(JSON)

    # Recommendations
    recommendations = Column(JSON)
    priority_actions = Column(JSON)
    follow_up_required = Column(Boolean, default=False)
    specialist_referral = Column(String(500))

    # Quality metrics
    confidence_score = Column(Float)
    processing_time = Column(Float)
    analysis_metadata = Column(JSON)

    # Cost tracking
    api_cost = Column(Float)
    token_usage = Column(JSON)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    patient = relationship("Patient", back_populates="analyses")


class AnalysisAuditLog(Base):
    """Audit trail for compliance"""
    __tablename__ = "analysis_audit_logs"

    id = Column(String(36), primary_key=True)
    analysis_id = Column(String(36), ForeignKey("multimodal_analyses.id"), nullable=False)

    action = Column(String(100), nullable=False)
    user_id = Column(String(100))
    ip_address = Column(String(50))
    user_agent = Column(String(500))

    details = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class APIUsageMetrics(Base):
    """Track API usage and costs"""
    __tablename__ = "api_usage_metrics"

    id = Column(String(36), primary_key=True)

    api_provider = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)

    # Request details
    operation_type = Column(String(100))
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    image_count = Column(Integer, default=0)

    # Cost
    cost = Column(Float)

    # Performance
    latency = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class SimilarCaseCache(Base):
    """Cache for similar case retrievals"""
    __tablename__ = "similar_case_cache"

    id = Column(String(36), primary_key=True)
    query_embedding = Column(Vector(1536))
    modality = Column(Enum(ModalityType))

    # Cached results
    similar_cases = Column(JSON)

    # Cache management
    cache_hits = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index('idx_similar_case_embedding', 'query_embedding', postgresql_using='ivfflat'),
    )
