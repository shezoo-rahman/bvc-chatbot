from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    finnhub_api_key: str = ""
    openai_model: str = "gpt-4o"

    model_config = SettingsConfigDict(env_file=".env")


settings: Settings = Settings()
