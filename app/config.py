"""Settings read from environment variables (or a .env file)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RotaTerra API"
    # Comma-separated list of front-end origins allowed to call the API
    cors_origins: str = "http://localhost:5173"
    http_timeout: int = 60
    cache_ttl_seconds: int = 86400

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
