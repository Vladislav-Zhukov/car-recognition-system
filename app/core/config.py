from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6380/0"

    class Config:
        env_file = ".env"


settings = Settings()