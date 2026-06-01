from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    finnhub_api_key: str = ""
    openai_model: str = "gpt-4o"

    model_config = {"env_file": ".env"}


settings = Settings()
