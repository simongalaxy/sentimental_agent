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

    # Processed data storage location.
    processed_files_folderPath: str
   
    # HuggingFace settings for ReviewsClassifer.
    hf_token: str

    # 1. DuplicatesRemover.
    duplicates_embedding_model: str

    # 2. SentimentalClassifier.
    sentimental_task: str
    sentimental_model: str

    # 3. TopicIdentifier.
    topic_embedding_model: str

    # ollama cloud settings for insights and report generation.
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
