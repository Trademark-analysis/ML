# final_pipeline.py
# top100 후보에 대해 OCR/벡터 생성 + test 이미지와 유사도 계산 + 최종 JSON 저장
# 최종 JSON에는 image_vector, text_vector 저장하지 않음

import os
import json
import time
import glob
import shutil
import requests
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

# =========================
# Azure 설정
# =========================

AZURE_VISION_KEY = os.getenv("SMU_VISION_KEY")
AZURE_VISION_ENDPOINT = os.getenv("SMU_VISION_ENDPOINT")

API_VERSION = "2024-02-01"
MODEL_VERSION = "2023-04-15"


# =========================
# 경로 설정
# =========================

# 1차 결과 JSON
INPUT_JSON = "../first_step/result/top100.json"

# top100 후보 이미지 폴더
DB_IMAGE_DIR = "../first_step/result/top100_images"

# 검색 이미지 폴더
QUERY_IMAGE_DIR = "test"

# 결과 저장 폴더
RESULT_DIR = "result"
RESULT_IMAGE_BASE_DIR = os.path.join(RESULT_DIR, "top10_images")

# 최종 결과 JSON
FINAL_RESULT_JSON = os.path.join(RESULT_DIR, "top10.json")

TOP_K = 10

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_IMAGE_BASE_DIR, exist_ok=True)


# =========================
# 기본 유틸
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_env():
    if not AZURE_VISION_KEY:
        raise ValueError("SMU_VISION_KEY가 .env에 없습니다.")
    if not AZURE_VISION_ENDPOINT:
        raise ValueError("SMU_VISION_ENDPOINT가 .env에 없습니다.")


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


# =========================
# Azure OCR / Vector
# =========================

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


# =========================
# 유사도 계산
# =========================

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return None

    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return None

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_type(has_image, has_text):
    if has_image and has_text:
        return "mixed"
    if has_image:
        return "image_only"
    if has_text:
        return "text_only"
    return "unknown"


# =========================
# 이미지 경로 처리
# =========================

def get_query_images(query_dir):
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(query_dir, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(query_dir, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(query_dir, "*.png")))
    return image_paths


def resolve_query_image_path(test_image_data):
    """
    top100.json의 test_image.image를 검색 이미지로 사용한다.
    test 폴더에서 따로 이미지를 찾지 않는다.
    """

    image_path = test_image_data.get("image")

    if not image_path:
        return None

    image_path = os.path.normpath(image_path)

    if os.path.exists(image_path):
        return image_path

    print(f"test_image에 적힌 이미지 경로가 존재하지 않습니다: {image_path}")
    return None

# =========================
# 후보 OCR + 벡터 생성
# =========================

def vectorize_candidate_items(candidate_items):
    vectorized_items = []

    print(f"\n후보 OCR/벡터 생성 시작: {len(candidate_items)}개")

    for item in tqdm(candidate_items):
        file_name = item.get("fileName")

        if not file_name:
            print("fileName 없음")
            continue

        image_path = os.path.join(DB_IMAGE_DIR, file_name)
        has_image_file = os.path.exists(image_path)

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

        trademark_type = get_type(has_image, has_text)

        new_item = {
            **item,
            "image_path": image_path if has_image_file else None,
            "ocr_text": ocr_text,
            "has_image": has_image,
            "has_text": has_text,
            "trademark_type": trademark_type,

            # 내부 계산용. 최종 JSON에는 저장하지 않음.
            "image_vector": image_vector,
            "text_vector": text_vector
        }

        vectorized_items.append(new_item)

        time.sleep(0.2)

    return vectorized_items


# =========================
# 유사 상표 Top10 검색
# =========================

def search_similar_trademarks(query_image_path, vectorized_items):
    query_name = os.path.splitext(os.path.basename(query_image_path))[0]

    print(f"\n검색 이미지: {query_image_path}")

    query_ocr_text = azure_ocr(query_image_path)
    query_image_vector = azure_image_vector(query_image_path)
    query_text_vector = azure_text_vector(query_ocr_text) if query_ocr_text else None

    query_has_image = query_image_vector is not None
    query_has_text = query_text_vector is not None
    query_type = get_type(query_has_image, query_has_text)

    similarity_results = []

    for item in vectorized_items:
        db_image_vector = item.get("image_vector")
        db_text_vector = item.get("text_vector")

        image_similarity = None
        text_similarity = None

        if query_has_image and db_image_vector is not None:
            image_similarity = cosine_similarity(query_image_vector, db_image_vector)

        if query_has_text and db_text_vector is not None:
            text_similarity = cosine_similarity(query_text_vector, db_text_vector)

        if image_similarity is not None and text_similarity is not None:
            final_score = 0.5 * image_similarity + 0.5 * text_similarity
            result_type = "mixed"
        elif image_similarity is not None:
            final_score = image_similarity
            result_type = "image_only"
        elif text_similarity is not None:
            final_score = text_similarity
            result_type = "text_only"
        else:
            continue

        # 최종 결과에는 vector 값 절대 넣지 않음
        result_item = {
            "rank": None,
            "match_group": item.get("match_group"),
            "fileName": item.get("fileName"),

            "ocr_text": item.get("ocr_text", ""),
            "type": result_type,

            "image_similarity": image_similarity,
            "text_similarity": text_similarity,
            "final_score": final_score,

            "vienna_codes": item.get("vienna_codes", []),
            "nice_codes": item.get("nice_codes", []),
            "matched_vienna_codes": item.get("matched_vienna_codes", []),
            "matched_nice_codes": item.get("matched_nice_codes", []),
        }

        similarity_results.append(result_item)

    similarity_results.sort(key=lambda x: x["final_score"], reverse=True)
    top10 = similarity_results[:TOP_K]

    for idx, item in enumerate(top10, start=1):
        item["rank"] = idx

    copy_top10_images(query_name, top10)

    return {
        "query_ocr_text": query_ocr_text,
        "query_type": query_type,
        "total_candidates": len(similarity_results),
        "top_k": TOP_K,
        "similar_trademark": top10
    }


# =========================
# Top10 이미지 복사
# =========================

def copy_top10_images(query_name, top10):
    result_image_dir = os.path.join(RESULT_IMAGE_BASE_DIR, query_name)
    os.makedirs(result_image_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for item in top10:
        file_name = item.get("fileName")

        if not file_name:
            missing_count += 1
            continue

        src_path = os.path.join(DB_IMAGE_DIR, file_name)
        dst_path = os.path.join(result_image_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"이미지 없음: {src_path}")
            missing_count += 1

    print(f"\nTop10 이미지 저장 폴더: {result_image_dir}")
    print(f"이미지 복사 성공: {copied_count}")
    print(f"이미지 없음: {missing_count}")


# =========================
# Main
# =========================

def main():
    check_env()

    raw_data = load_json(INPUT_JSON)

    test_image_data = raw_data.get("test_image", {})
    candidate_items = raw_data.get("results", [])

    if not candidate_items:
        print("top100.json 안에 results 후보가 없습니다.")
        return

    query_image_path = resolve_query_image_path(test_image_data)

    if not query_image_path:
        print("검색할 test 이미지가 없습니다.")
        return

    # 1. 후보 OCR + 벡터 생성
    vectorized_items = vectorize_candidate_items(candidate_items)

    # 2. test 이미지와 후보들 유사도 계산
    search_result = search_similar_trademarks(query_image_path, vectorized_items)

    # 3. 최종 JSON 구성
    final_output = {
        "image": test_image_data.get("image"),
        "service_description": test_image_data.get("service_description"),
        "nice_codes": test_image_data.get("nice_codes", []),
        "similar_group_codes": test_image_data.get("similar_group_codes", []),
        "vienna_codes": test_image_data.get("vienna_codes", []),
        "similar_trademark": search_result.get("similar_trademark", [])
    }

    save_json(final_output, FINAL_RESULT_JSON)

    print(f"\n최종 JSON 저장 완료: {FINAL_RESULT_JSON}")
    print("===== 전체 완료 =====")


if __name__ == "__main__":
    main()