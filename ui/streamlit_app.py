"""
Streamlit Multi-Modal Medical Analysis Dashboard
Complete UI for healthcare AI analysis
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import sys
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.multimodal.processors.ecg_processor import ECGProcessor
from src.multimodal.processors.clinical_processor import ClinicalNotesProcessor
from src.multimodal.analyzers.multimodal_analyzer import MultiModalAnalyzer
from src.multimodal.analyzers.rag_system import MultiModalRAG, MultiModalOrchestrator
from src.database.database import db_manager

# Page config
st.set_page_config(
    page_title="DrBinati MedAssist - AI Medical Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1rem 0;
    }
    .brand-logo {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.5rem 0;
        letter-spacing: -1px;
    }
    .brand-tagline {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        font-weight: 400;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .sidebar-brand {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sidebar-brand h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sidebar-brand p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = "ANON_" + str(hash(str(pd.Timestamp.now())))[:8]
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'cost_tracker' not in st.session_state:
    st.session_state.cost_tracker = {'total': 0.0, 'operations': []}


def display_header():
    """Display main header"""
    st.markdown('<div class="brand-logo">🏥 DrBinati MedAssist</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">AI-Powered Multi-Modal Medical Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")


def display_sidebar():
    """Display sidebar with configuration"""
    with st.sidebar:
        # Brand header in sidebar
        st.markdown("""
        <div class="sidebar-brand">
            <h1>🏥 DrBinati<br/>MedAssist</h1>
            <p>AI Medical Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Configuration")
        
        # API Selection
        api_choice = st.selectbox(
            "AI Provider",
            ["anthropic", "openai"],
            index=0 if settings.api_choice == "anthropic" else 1,
            help="Choose between Claude (Anthropic) or GPT-4 (OpenAI)"
        )
        
        # Patient ID
        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
            help="Anonymous patient identifier"
        )
        st.session_state.patient_id = patient_id
        
        # Analysis Settings
        st.markdown("### 🎯 Analysis Settings")
        
        use_rag = st.checkbox(
            "Use Similar Cases (RAG)",
            value=True,
            help="Retrieve similar cases for context"
        )
        
        condition_focus = st.selectbox(
            "Condition Focus",
            ["None", "Acute MI", "Arrhythmia", "Heart Failure", "Oncology"],
            help="Focus analysis on specific condition"
        )
        
        # Cost Tracker
        st.markdown("### 💰 Cost Tracker")
        st.metric("Session Cost", f"${st.session_state.cost_tracker['total']:.4f}")
        
        if st.button("Reset Session"):
            st.session_state.cost_tracker = {'total': 0.0, 'operations': []}
            st.rerun()
        
        return api_choice, use_rag, condition_focus


def upload_section():
    """File upload section"""
    st.markdown('<div class="sub-header">📁 Upload Medical Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 ECG Signal")
        ecg_file = st.file_uploader(
            "Upload ECG",
            type=['dat', 'csv', 'npy', 'txt'],
            help="Supported: WFDB, CSV, NumPy"
        )
    
    with col2:
        st.markdown("#### 🔬 Medical Imaging")
        image_file = st.file_uploader(
            "Upload Image",
            type=['dcm', 'png', 'jpg', 'jpeg'],
            help="Supported: DICOM, PNG, JPEG"
        )
        
        if image_file:
            modality = st.selectbox(
                "Imaging Modality",
                ["XR", "CT", "MR", "US"],
                help="Select imaging modality"
            )
    
    with col3:
        st.markdown("#### 📝 Clinical Notes")
        notes_file = st.file_uploader(
            "Upload Clinical Notes",
            type=['txt', 'docx', 'pdf'],
            help="Supported: TXT, DOCX, PDF"
        )
    
    return ecg_file, image_file, notes_file, modality if image_file else None


def process_files(ecg_file, image_file, notes_file, modality):
    """Process uploaded files"""
    processed_data = {}
    
    with st.spinner("Processing files..."):
        # Process ECG
        if ecg_file:
            try:
                # Save temporarily
                temp_path = Path(tempfile.gettempdir()) / ecg_file.name
                with open(temp_path, 'wb') as f:
                    f.write(ecg_file.getvalue())
                
                processor = ECGProcessor()
                ecg_signal, metadata = processor.load_ecg_file(temp_path)
                processed_ecg = processor.process_ecg(ecg_signal)
                processed_ecg['file_path'] = str(temp_path)
                processed_data['ecg'] = processed_ecg
                
                st.success("✅ ECG processed successfully")
            except Exception as e:
                st.error(f"❌ ECG processing failed: {e}")
        
        # Process Image
        if image_file:
            try:
                temp_path = Path(tempfile.gettempdir()) / image_file.name
                with open(temp_path, 'wb') as f:
                    f.write(image_file.getvalue())

                # Load image with PIL
                img = Image.open(temp_path)

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Convert to base64 for display
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                # Create processed image structure
                processed_image = {
                    'file_path': str(temp_path),
                    'modality': modality,
                    'image_base64': img_base64,
                    'metadata_text': f"Modality: {modality}\nSize: {img.size}\nMode: {img.mode}"
                }
                processed_data['imaging'] = processed_image

                st.success("✅ Medical image processed successfully")
            except Exception as e:
                st.error(f"❌ Image processing failed: {e}")
        
        # Process Clinical Notes
        if notes_file:
            try:
                notes_text = notes_file.read().decode('utf-8')
                processor = ClinicalNotesProcessor()
                processed_notes = processor.process_notes(notes_text)
                processed_notes['file_path'] = notes_file.name
                processed_data['notes'] = processed_notes
                
                st.success("✅ Clinical notes processed successfully")
            except Exception as e:
                st.error(f"❌ Notes processing failed: {e}")
    
    return processed_data


def display_processed_data(processed_data):
    """Display processed data"""
    if not processed_data:
        return
    
    st.markdown('<div class="sub-header">👁️ Processed Data Preview</div>', unsafe_allow_html=True)
    
    tabs = st.tabs([k.upper() for k in processed_data.keys()])
    
    for i, (key, data) in enumerate(processed_data.items()):
        with tabs[i]:
            if key == 'ecg':
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**ECG Waveform**")
                    # Display waveform image
                    img_bytes = base64.b64decode(data['waveform_image_base64'])
                    st.image(img_bytes, width='stretch')
                
                with col2:
                    st.markdown("**Measurements**")
                    st.code(data['annotations'], language='text')
            
            elif key == 'imaging':
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Medical Image**")
                    img_bytes = base64.b64decode(data['image_base64'])
                    st.image(img_bytes, width='stretch')
                
                with col2:
                    st.markdown("**Metadata**")
                    st.code(data['metadata_text'], language='text')
            
            elif key == 'notes':
                st.markdown("**Structured Clinical Notes**")
                for section, content in data['structured'].items():
                    if content:
                        with st.expander(f"📄 {section.replace('_', ' ').title()}"):
                            st.write(content)


def run_analysis(processed_data, api_choice, use_rag, condition_focus):
    """Run multi-modal analysis"""
    if not processed_data:
        st.warning("⚠️ Please upload at least one file to analyze")
        return None
    
    st.markdown('<div class="sub-header">🤖 AI Analysis Results</div>', unsafe_allow_html=True)
    
    # Map condition focus
    condition_map = {
        "None": None,
        "Acute MI": "acute_mi",
        "Arrhythmia": "arrhythmia",
        "Heart Failure": None,
        "Oncology": "oncology"
    }
    condition = condition_map.get(condition_focus)
    
    with st.spinner("🔄 Analyzing with AI... This may take 30-60 seconds..."):
        try:
            # Initialize analyzer
            analyzer = MultiModalAnalyzer(api_choice=api_choice)
            
            results = {}
            
            # Analyze ECG
            if 'ecg' in processed_data:
                with st.status("Analyzing ECG..."):
                    ecg_result = analyzer.analyze_ecg(
                        processed_data['ecg'],
                        condition_focus=condition
                    )
                    results['ecg'] = ecg_result
            
            # Analyze Imaging
            if 'imaging' in processed_data:
                with st.status("Analyzing medical imaging..."):
                    imaging_result = analyzer.analyze_medical_image(
                        processed_data['imaging'],
                        modality=processed_data['imaging']['modality'],
                        focus=condition
                    )
                    results['imaging'] = imaging_result
            
            # Analyze Clinical Notes
            if 'notes' in processed_data:
                with st.status("Analyzing clinical notes..."):
                    notes_result = analyzer.analyze_clinical_notes(
                        processed_data['notes'],
                        specialty="cardiology" if condition in ['acute_mi', 'arrhythmia'] else None
                    )
                    results['clinical'] = notes_result
            
            # Synthesize
            with st.status("Synthesizing multi-modal assessment..."):
                synthesis = analyzer.synthesize_multimodal(
                    ecg_analysis=results.get('ecg', {}).get('analysis', 'N/A'),
                    imaging_analysis=results.get('imaging', {}).get('analysis', 'N/A'),
                    clinical_analysis=results.get('clinical', {}).get('analysis', 'N/A'),
                    context='general'
                )
                results['synthesis'] = synthesis
            
            # Update cost tracker
            cost_summary = analyzer.get_cost_summary()
            st.session_state.cost_tracker['total'] += cost_summary['total_cost']
            st.session_state.cost_tracker['operations'].extend(cost_summary['operations'])
            
            return results
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            return None


def display_results(results):
    """Display analysis results"""
    if not results:
        return
    
    # Create tabs for different analyses
    tab_names = []
    if 'ecg' in results:
        tab_names.append("📊 ECG Analysis")
    if 'imaging' in results:
        tab_names.append("🔬 Imaging Analysis")
    if 'clinical' in results:
        tab_names.append("📝 Clinical Analysis")
    if 'synthesis' in results:
        tab_names.append("🎯 Comprehensive Assessment")
    
    tabs = st.tabs(tab_names)
    
    tab_idx = 0
    
    # ECG Analysis
    if 'ecg' in results:
        with tabs[tab_idx]:
            st.markdown("### ECG Interpretation")
            st.markdown(results['ecg']['analysis'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{results['ecg']['latency']:.2f}s")
            with col2:
                st.metric("Tokens Used", 
                         f"{results['ecg']['tokens']['input'] + results['ecg']['tokens']['output']}")
        tab_idx += 1
    
    # Imaging Analysis
    if 'imaging' in results:
        with tabs[tab_idx]:
            st.markdown("### Radiology Report")
            st.markdown(results['imaging']['analysis'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{results['imaging']['latency']:.2f}s")
            with col2:
                st.metric("Tokens Used", 
                         f"{results['imaging']['tokens']['input'] + results['imaging']['tokens']['output']}")
        tab_idx += 1
    
    # Clinical Analysis
    if 'clinical' in results:
        with tabs[tab_idx]:
            st.markdown("### Clinical Assessment")
            st.markdown(results['clinical']['analysis'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{results['clinical']['latency']:.2f}s")
            with col2:
                st.metric("Tokens Used", 
                         f"{results['clinical']['tokens']['input'] + results['clinical']['tokens']['output']}")
        tab_idx += 1
    
    # Comprehensive Synthesis
    if 'synthesis' in results:
        with tabs[tab_idx]:
            st.markdown("### Integrated Multi-Modal Assessment")
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(results['synthesis']['synthesis'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{results['synthesis']['latency']:.2f}s")
            with col2:
                st.metric("Tokens Used", 
                         f"{results['synthesis']['tokens']['input'] + results['synthesis']['tokens']['output']}")


def display_cost_breakdown():
    """Display cost breakdown"""
    if st.session_state.cost_tracker['operations']:
        st.markdown('<div class="sub-header">💳 Cost Breakdown</div>', unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.cost_tracker['operations'])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", f"${df['cost'].sum():.4f}")
        with col2:
            st.metric("Total Input Tokens", f"{df['input_tokens'].sum():,}")
        with col3:
            st.metric("Total Output Tokens", f"{df['output_tokens'].sum():,}")
        with col4:
            st.metric("Avg Latency", f"{df['latency'].mean():.2f}s")
        
        # Cost by operation
        fig = px.bar(
            df.groupby('operation')['cost'].sum().reset_index(),
            x='operation',
            y='cost',
            title='Cost by Operation Type',
            labels={'cost': 'Cost ($)', 'operation': 'Operation'}
        )
        st.plotly_chart(fig, width='stretch')


def main():
    """Main application"""
    display_header()
    
    # Sidebar
    api_choice, use_rag, condition_focus = display_sidebar()
    
    # Main content
    st.markdown('<div class="info-box">⚕️ Upload medical data for comprehensive AI-powered analysis using state-of-the-art multimodal models.</div>', unsafe_allow_html=True)
    
    # File upload
    ecg_file, image_file, notes_file, modality = upload_section()
    
    # Process button
    if st.button("🚀 Process Files", type="primary", width='stretch'):
        processed_data = process_files(ecg_file, image_file, notes_file, modality)
        st.session_state['processed_data'] = processed_data

    # Display processed data if available
    if 'processed_data' in st.session_state and st.session_state['processed_data']:
        display_processed_data(st.session_state['processed_data'])
        
        # Analyze button
        if st.button("🤖 Run AI Analysis", type="primary", width='stretch'):
            results = run_analysis(
                st.session_state['processed_data'],
                api_choice,
                use_rag,
                condition_focus
            )
            
            if results:
                st.session_state['analysis_results'] = results
    
    # Display results if available
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        display_results(st.session_state['analysis_results'])
        
        # Export option
        st.markdown("---")
        if st.button("📥 Export Report", width='stretch'):
            # Create report
            report = "# Multi-Modal Medical Analysis Report\n\n"
            report += f"**Patient ID:** {st.session_state.patient_id}\n\n"
            
            for key, result in st.session_state['analysis_results'].items():
                if key == 'synthesis':
                    report += "## Comprehensive Assessment\n\n"
                    report += result['synthesis'] + "\n\n"
                else:
                    report += f"## {key.upper()} Analysis\n\n"
                    report += result['analysis'] + "\n\n"
            
            st.download_button(
                label="Download Report (Markdown)",
                data=report,
                file_name=f"medical_report_{st.session_state.patient_id}.md",
                mime="text/markdown"
            )
    
    # Cost breakdown
    display_cost_breakdown()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p><strong>DrBinati MedAssist</strong> | Powered by Claude & GPT-4 Vision | © 2025 Binati AInalytics</p>
        <p><small>⚠️ For research and educational purposes only. Not for clinical use.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
