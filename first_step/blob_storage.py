import os
import uuid
from typing import BinaryIO

from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "trademark-logos")


def upload_image_to_blob(
    file: BinaryIO,
    original_filename: str,
    content_type: str = "image/png",
) -> str:
    """
    사용자가 업로드한 이미지 파일을 Azure Blob Storage에 업로드하고,
    접근 가능한 blob URL 반환
    """

    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING 환경변수가 없습니다.")

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)

    # 컨테이너가 없으면 생성
    try:
        container_client.create_container()
    except Exception:
        pass

    ext = os.path.splitext(original_filename)[1] or ".png"
    blob_name = f"logos/{uuid.uuid4()}{ext}"

    blob_client = container_client.get_blob_client(blob_name)

    blob_client.upload_blob(
        file,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )

    return blob_client.url