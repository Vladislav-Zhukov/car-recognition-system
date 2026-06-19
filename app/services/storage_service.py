import uuid
from io import BytesIO
from pathlib import Path

import boto3

from app.core.config import settings


class StorageService:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
        )
        self.bucket_name = settings.S3_BUCKET_NAME

    def upload_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        ext = Path(filename).suffix.lower()
        object_key = f"uploads/{uuid.uuid4()}{ext}"

        self.client.upload_fileobj(
            BytesIO(file_bytes),
            self.bucket_name,
            object_key,
            ExtraArgs={"ContentType": content_type or "image/jpeg"},
        )

        return object_key


storage_service = StorageService()