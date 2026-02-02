"""
This module is used to load the environment variables from the .env file.
It uses the pydantic_settings library to load the environment variables.
The purpose of this module is to load the environment variables from the .env file and make them available to the rest of the application.
In case there is not a value for an environment variable, the module will raise a ValueError.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

config = Config()