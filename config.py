from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    meta_verify_token: str = ""
    meta_app_secret: str = ""
    meta_phone_id: str = ""
    meta_access_token: str = ""
    chutes_api_key: str = ""
    database_url: str = "sqlite:///./treasury.db"
    env: str = "dev"


settings = Settings()
