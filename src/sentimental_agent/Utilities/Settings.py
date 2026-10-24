import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    
    # logger settings.
    log_path: str
    log_level: str

    # summary report settings.
    summary_report_path: str

    # Data Source configurations.
    source_filename: str
    source_folderpath: str

    # Temporary files configurations.
    tempfiles_folderpath: str

    # Processed Data configurations.
    processed_files_folderPath: str

    # HuggingFace settings.
    # 1. sentimental model.
    sentimental_model: str
    categorizier_model: str
    labeller_model: str

    # ollama cloud settings.
    cloud_model: str
    cloud_api_key: str
    cloud_host: str

    # ollama local settings.
    local_model: str
    local_host: str

    # pydantic settings config.
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    

# Singleton instance of the settings.
settings = Settings() # type: ignore
