"""
Application configuration using Pydantic Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    api_choice: str = "anthropic"
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # Database Configuration
    database_url: str = "postgresql://postgres:postgres@localhost:5432/multimodal_healthcare"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "multimodal_healthcare"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    
    # Vector Database
    vector_dimension: int = 1536
    
    # Application Settings
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Storage
    upload_dir: Path = Path("./data/uploads")
    processed_dir: Path = Path("./data/processed")
    max_upload_size: int = 50  # MB
    
    # API Rate Limiting
    rate_limit_per_minute: int = 60
    max_concurrent_requests: int = 10
    
    # Cost Management
    daily_api_budget: float = 50.0
    monthly_api_budget: float = 1000.0
    alert_threshold: float = 0.8
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra='ignore'  # Ignore extra environment variables not defined in Settings
    )
    
    def get_database_url(self) -> str:
        """Get formatted database URL"""
        return self.database_url
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()


# Model configurations
MODEL_CONFIGS = {
    "anthropic": {
        "claude-3-5-haiku-20241022": {
            "max_tokens": 8192,
            "supports_vision": True,
            "pricing": {
                "input": 1.00 / 1_000_000,   # per token (cheaper)
                "output": 5.00 / 1_000_000,
                "image": 0.40 / 1000          # per image
            }
        },
        "claude-sonnet-4-5-20250929": {
            "max_tokens": 8192,
            "supports_vision": True,
            "pricing": {
                "input": 3.00 / 1_000_000,   # per token
                "output": 15.00 / 1_000_000,
                "image": 4.80 / 1000          # per image
            }
        },
        "claude-3-7-sonnet-latest": {
            "max_tokens": 8192,
            "supports_vision": True,
            "pricing": {
                "input": 3.00 / 1_000_000,   # per token
                "output": 15.00 / 1_000_000,
                "image": 4.80 / 1000          # per image
            }
        },
        "claude-3-opus-20240229": {
            "max_tokens": 4096,
            "supports_vision": True,
            "pricing": {
                "input": 15.00 / 1_000_000,
                "output": 75.00 / 1_000_000,
                "image": 24.00 / 1000
            }
        }
    },
    "openai": {
        "gpt-4o": {
            "max_tokens": 4096,
            "supports_vision": True,
            "pricing": {
                "input": 2.50 / 1_000_000,
                "output": 10.00 / 1_000_000,
                "image": 0.00255
            }
        },
        "gpt-4-turbo": {
            "max_tokens": 4096,
            "supports_vision": True,
            "pricing": {
                "input": 10.00 / 1_000_000,
                "output": 30.00 / 1_000_000,
                "image": 0.00765
            }
        }
    }
}


# Medical modality configurations
MODALITY_CONFIGS = {
    "ecg": {
        "supported_formats": [".dat", ".edf", ".txt", ".csv"],
        "sampling_rates": [250, 500, 1000],  # Hz
        "num_leads": [1, 3, 12],
        "max_duration": 600  # seconds
    },
    "imaging": {
        "supported_formats": [".dcm", ".nii", ".nii.gz", ".png", ".jpg"],
        "modalities": ["CR", "CT", "MR", "US", "XA", "DX"],
        "max_size": 512 * 1024 * 1024,  # 512 MB
        "target_resolution": (512, 512)
    },
    "clinical_notes": {
        "max_length": 50000,  # characters
        "required_sections": ["chief_complaint", "history", "assessment"],
        "supported_formats": [".txt", ".docx", ".pdf"]
    }
}
