# pip install requests python-dotenv tqdm

import os
import json
import time
import random
import requests
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

IMAGE_DIR = "image_filtered"
INPUT_JSON = "filtered_niceCode.json"
OUTPUT_JSON = "trademark_vector.json"

SAMPLE_SIZE = 500
RANDOM_SEED = 42

API_VERSION = "2024-02-01"
MODEL_VERSION = "2023-04-15"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_env():
    if not AZURE_VISION_KEY:
        raise ValueError("AZURE_VISION_KEY가 .env에 없습니다.")
    if not AZURE_VISION_ENDPOINT:
        raise ValueError("AZURE_VISION_ENDPOINT가 .env에 없습니다.")


def request_with_retry(url, headers, data=None, json_body=None, max_retry=3):
    for attempt in range(max_retry):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                json=json_body,
                timeout=300
            )

            if response.status_code == 200:
                return response.json()

            print(f"요청 실패 {response.status_code}: {response.text}")

        except Exception as e:
            print(f"재시도 {attempt + 1}/{max_retry} 실패: {e}")

        time.sleep(5)

    return None


def azure_ocr(image_path):
    url = (
        AZURE_VISION_ENDPOINT.rstrip("/")
        + "/computervision/imageanalysis:analyze"
        + "?api-version=2024-02-01"
        + "&features=read"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_VISION_KEY,
        "Content-Type": "application/octet-stream"
    }

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = request_with_retry(url, headers, data=image_bytes)

    if result is None:
        return ""

    lines = []
    blocks = result.get("readResult", {}).get("blocks", [])

    for block in blocks:
        for line in block.get("lines", []):
            text = line.get("text", "").strip()
            if text:
                lines.append(text)

    return " ".join(lines).strip()


def azure_image_vector(image_path):
    url = (
        AZURE_VISION_ENDPOINT.rstrip("/")
        + "/computervision/retrieval:vectorizeImage"
        + f"?api-version={API_VERSION}&model-version={MODEL_VERSION}"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_VISION_KEY,
        "Content-Type": "application/octet-stream"
    }

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = request_with_retry(url, headers, data=image_bytes)

    if result is None:
        return None

    return result.get("vector")


def azure_text_vector(text):
    if not text or not text.strip():
        return None

    url = (
        AZURE_VISION_ENDPOINT.rstrip("/")
        + "/computervision/retrieval:vectorizeText"
        + f"?api-version={API_VERSION}&model-version={MODEL_VERSION}"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_VISION_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "text": text
    }

    result = request_with_retry(url, headers, json_body=body)

    if result is None:
        return None

    return result.get("vector")


def get_trademark_type(has_image, has_text):
    if has_text and not has_image:
        return "text_only"

    if has_image and not has_text:
        return "image_only"

    if has_image and has_text:
        return "mixed"

    return "unknown"


def main():
    check_env()

    raw_data = load_json(INPUT_JSON)
    all_data = raw_data["images"]

    random.seed(RANDOM_SEED)

    if len(all_data) > SAMPLE_SIZE:
        data = random.sample(all_data, SAMPLE_SIZE)
    else:
        data = all_data

    result = []

    print(f"전체 데이터 수: {len(all_data)}")
    print(f"이번에 처리할 데이터 수: {len(data)}")

    for item in tqdm(data):
        file_name = item.get("fileName")
        image_path = os.path.join(IMAGE_DIR, file_name) if file_name else None

        has_image_file = image_path is not None and os.path.exists(image_path)

        ocr_text = ""
        image_vector = None
        text_vector = None

        if has_image_file:
            ocr_text = azure_ocr(image_path)
            image_vector = azure_image_vector(image_path)
        else:
            print(f"이미지 없음: {image_path}")

        has_text = bool(ocr_text and ocr_text.strip())
        has_image = image_vector is not None

        if has_text:
            text_vector = azure_text_vector(ocr_text)

        trademark_type = get_trademark_type(
            has_image=has_image,
            has_text=has_text
        )

        new_item = {
            **item,
            "image_path": image_path if has_image_file else None,
            "ocr_text": ocr_text,
            "has_image": has_image,
            "has_text": has_text,
            "trademark_type": trademark_type,
            "image_vector": image_vector,
            "text_vector": text_vector
        }

        result.append(new_item)

        time.sleep(0.2)

    save_json(result, OUTPUT_JSON)

    print(f"저장 완료: {OUTPUT_JSON}")
    print(f"총 저장 개수: {len(result)}")


if __name__ == "__main__":
    main()