from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6380/0"
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minio"
    S3_SECRET_KEY: str = "minio12345"
    S3_BUCKET_NAME: str = "car-images"

    class Config:
        env_file = ".env"


settings = Settings()