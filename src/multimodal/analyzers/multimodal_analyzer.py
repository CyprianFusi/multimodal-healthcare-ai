"""
Multi-Modal Medical Analyzers
Uses Claude or GPT-4 Vision APIs for analysis
"""
from typing import Dict, Optional, List
import logging
from anthropic import Anthropic
from openai import OpenAI
import os
import time
from config.settings import settings, MODEL_CONFIGS
from src.multimodal.prompts.medical_prompts import PromptLibrary

logger = logging.getLogger(__name__)


class CostTracker:
    """Track API usage and costs"""
    
    def __init__(self):
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images = 0
    
    def log_usage(
        self,
        operation: str,
        api_provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        images: int = 0,
        latency: float = 0
    ):
        """Log API usage"""
        pricing = MODEL_CONFIGS[api_provider][model]['pricing']
        
        cost = (
            input_tokens * pricing['input'] +
            output_tokens * pricing['output'] +
            images * pricing['image']
        )
        
        self.session_costs.append({
            'operation': operation,
            'api_provider': api_provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'images': images,
            'cost': cost,
            'latency': latency
        })
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_images += images
    
    def get_session_summary(self) -> Dict:
        """Get session cost summary"""
        total_cost = sum(item['cost'] for item in self.session_costs)
        avg_latency = sum(item['latency'] for item in self.session_costs) / len(self.session_costs) if self.session_costs else 0
        
        return {
            'total_cost': total_cost,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_images': self.total_images,
            'average_latency': avg_latency,
            'operations': self.session_costs
        }


class MultiModalAnalyzer:
    """Analyze medical data using Claude or GPT-4 Vision"""
    
    def __init__(self, api_choice: str = None):
        """
        Initialize analyzer

        Args:
            api_choice: 'anthropic' or 'openai'. If None, uses settings default
        """
        self.api_choice = api_choice or settings.api_choice
        self.cost_tracker = CostTracker()

        if self.api_choice == 'anthropic':
            # Prioritize environment variable over settings (important for notebooks)
            api_key = os.getenv('ANTHROPIC_API_KEY') or settings.anthropic_api_key
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment or settings")
            self.client = Anthropic(api_key=api_key)
            # Use model from environment or default, strip any quotes
            model_name = os.getenv('CLAUDE_MODEL') or 'claude-3-7-sonnet-latest'
            self.model = model_name.strip("'\"")  # Remove quotes if present
            logger.info(f"Using Claude model: {self.model}")
        elif self.api_choice == 'openai':
            # Prioritize environment variable over settings
            api_key = os.getenv('OPENAI_API_KEY') or settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or settings")
            self.client = OpenAI(api_key=api_key)
            self.model = 'gpt-4o'
        else:
            raise ValueError(f"Invalid API choice: {self.api_choice}")

        logger.info(f"Initialized MultiModalAnalyzer with {self.api_choice} ({self.model})")
    
    def analyze_ecg(
        self,
        ecg_data: Dict,
        condition_focus: Optional[str] = None
    ) -> Dict:
        """
        Analyze ECG using vision + text
        
        Args:
            ecg_data: Dict with 'waveform_image_base64', 'annotations', 'metadata'
            condition_focus: Optional focus ('acute_mi', 'arrhythmia', None)
        
        Returns:
            Dict with 'analysis', 'metadata', 'cost'
        """
        try:
            start_time = time.time()
            
            # Get appropriate prompt
            prompt = PromptLibrary.get_ecg_prompt(
                condition=condition_focus,
                annotations=ecg_data['annotations'],
                metadata=ecg_data['metadata']
            )
            
            # Make API call with vision
            if self.api_choice == 'anthropic':
                response = self._call_claude_vision(
                    prompt=prompt,
                    image_base64=ecg_data['waveform_image_base64'],
                    max_tokens=2500
                )
                analysis = response['text']
                tokens = response['tokens']
            else:
                response = self._call_openai_vision(
                    prompt=prompt,
                    image_base64=ecg_data['waveform_image_base64'],
                    max_tokens=2500
                )
                analysis = response['text']
                tokens = response['tokens']
            
            latency = time.time() - start_time
            
            # Track costs
            self.cost_tracker.log_usage(
                operation='ecg_analysis',
                api_provider=self.api_choice,
                model=self.model,
                input_tokens=tokens['input'],
                output_tokens=tokens['output'],
                images=1,
                latency=latency
            )
            
            return {
                'analysis': analysis,
                'condition_focus': condition_focus,
                'latency': latency,
                'tokens': tokens
            }
            
        except Exception as e:
            logger.error(f"ECG analysis failed: {e}")
            raise
    
    def analyze_medical_image(
        self,
        image_data: Dict,
        modality: str,
        focus: Optional[str] = None
    ) -> Dict:
        """
        Analyze medical imaging
        
        Args:
            image_data: Dict with 'image_base64', 'metadata_text', 'metadata'
            modality: Image modality (XR, CT, MR, etc.)
            focus: Optional focus (e.g., 'oncology')
        
        Returns:
            Dict with 'analysis', 'metadata', 'cost'
        """
        try:
            start_time = time.time()
            
            # Get appropriate prompt
            prompt = PromptLibrary.get_imaging_prompt(
                modality=modality,
                focus=focus,
                metadata=image_data.get('metadata')
            )
            
            # Make API call
            if self.api_choice == 'anthropic':
                response = self._call_claude_vision(
                    prompt=prompt,
                    image_base64=image_data['image_base64'],
                    max_tokens=2500
                )
                analysis = response['text']
                tokens = response['tokens']
            else:
                response = self._call_openai_vision(
                    prompt=prompt,
                    image_base64=image_data['image_base64'],
                    max_tokens=2500
                )
                analysis = response['text']
                tokens = response['tokens']
            
            latency = time.time() - start_time
            
            # Track costs
            self.cost_tracker.log_usage(
                operation='imaging_analysis',
                api_provider=self.api_choice,
                model=self.model,
                input_tokens=tokens['input'],
                output_tokens=tokens['output'],
                images=1,
                latency=latency
            )
            
            return {
                'analysis': analysis,
                'modality': modality,
                'focus': focus,
                'latency': latency,
                'tokens': tokens
            }
            
        except Exception as e:
            logger.error(f"Imaging analysis failed: {e}")
            raise
    
    def analyze_clinical_notes(
        self,
        notes_data: Dict,
        specialty: Optional[str] = None
    ) -> Dict:
        """
        Analyze clinical documentation
        
        Args:
            notes_data: Processed notes dict with 'structured' and 'anonymized'
            specialty: Optional specialty focus ('cardiology', etc.)
        
        Returns:
            Dict with 'analysis', 'metadata', 'cost'
        """
        try:
            start_time = time.time()
            
            # Format notes for LLM
            from src.multimodal.processors.clinical_processor import ClinicalNotesProcessor
            processor = ClinicalNotesProcessor(use_presidio=False)
            formatted_notes = processor.format_for_llm(notes_data)
            
            # Get appropriate prompt
            prompt = PromptLibrary.get_clinical_notes_prompt(
                specialty=specialty,
                notes=formatted_notes
            )
            
            # Make API call (text only)
            if self.api_choice == 'anthropic':
                response = self._call_claude_text(
                    prompt=prompt,
                    max_tokens=2500
                )
                analysis = response['text']
                tokens = response['tokens']
            else:
                response = self._call_openai_text(
                    prompt=prompt,
                    max_tokens=2500
                )
                analysis = response['text']
                tokens = response['tokens']
            
            latency = time.time() - start_time
            
            # Track costs
            self.cost_tracker.log_usage(
                operation='clinical_notes_analysis',
                api_provider=self.api_choice,
                model=self.model,
                input_tokens=tokens['input'],
                output_tokens=tokens['output'],
                images=0,
                latency=latency
            )
            
            return {
                'analysis': analysis,
                'specialty': specialty,
                'latency': latency,
                'tokens': tokens
            }
            
        except Exception as e:
            logger.error(f"Clinical notes analysis failed: {e}")
            raise
    
    def synthesize_multimodal(
        self,
        ecg_analysis: str = "N/A",
        imaging_analysis: str = "N/A",
        clinical_analysis: str = "N/A",
        context: str = "general"
    ) -> Dict:
        """
        Synthesize all modalities into comprehensive assessment
        
        Args:
            ecg_analysis: ECG analysis text
            imaging_analysis: Imaging analysis text
            clinical_analysis: Clinical notes analysis text
            context: 'general' or 'emergency'
        
        Returns:
            Dict with 'synthesis', 'metadata', 'cost'
        """
        try:
            start_time = time.time()
            
            # Get synthesis prompt
            prompt = PromptLibrary.get_synthesis_prompt(
                context=context,
                ecg_analysis=ecg_analysis,
                imaging_analysis=imaging_analysis,
                clinical_analysis=clinical_analysis
            )
            
            # Make API call
            if self.api_choice == 'anthropic':
                response = self._call_claude_text(
                    prompt=prompt,
                    max_tokens=3500
                )
                synthesis = response['text']
                tokens = response['tokens']
            else:
                response = self._call_openai_text(
                    prompt=prompt,
                    max_tokens=3500
                )
                synthesis = response['text']
                tokens = response['tokens']
            
            latency = time.time() - start_time
            
            # Track costs
            self.cost_tracker.log_usage(
                operation='multimodal_synthesis',
                api_provider=self.api_choice,
                model=self.model,
                input_tokens=tokens['input'],
                output_tokens=tokens['output'],
                images=0,
                latency=latency
            )
            
            return {
                'synthesis': synthesis,
                'context': context,
                'latency': latency,
                'tokens': tokens
            }
            
        except Exception as e:
            logger.error(f"Multimodal synthesis failed: {e}")
            raise
    
    def _call_claude_vision(self, prompt: str, image_base64: str, max_tokens: int = 2048) -> Dict:
        """Call Claude with vision capability"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            return {
                'text': response.content[0].text,
                'tokens': {
                    'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def _call_claude_text(self, prompt: str, max_tokens: int = 2048) -> Dict:
        """Call Claude with text only"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return {
                'text': response.content[0].text,
                'tokens': {
                    'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def _call_openai_vision(self, prompt: str, image_base64: str, max_tokens: int = 2048) -> Dict:
        """Call GPT-4o with vision capability"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            return {
                'text': response.choices[0].message.content,
                'tokens': {
                    'input': response.usage.prompt_tokens,
                    'output': response.usage.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _call_openai_text(self, prompt: str, max_tokens: int = 2048) -> Dict:
        """Call GPT-4o with text only"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return {
                'text': response.choices[0].message.content,
                'tokens': {
                    'input': response.usage.prompt_tokens,
                    'output': response.usage.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def get_cost_summary(self) -> Dict:
        """Get session cost summary"""
        return self.cost_tracker.get_session_summary()
