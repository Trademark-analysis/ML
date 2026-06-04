# 검색 이미지와 DB 이미지 간의 유사도를 계산하여 상위 10개 결과를 JSON 파일로 저장

import os
import json
import glob
import shutil
import numpy as np
from dotenv import load_dotenv

from ocr import azure_ocr, azure_image_vector, azure_text_vector, check_env

load_dotenv()

VECTOR_JSON_PATH = "trademark_vector.json"
DB_IMAGE_DIR = "image_filtered"

QUERY_IMAGE_DIR = "test"

RESULT_DIR = "result"
RESULT_IMAGE_BASE_DIR = os.path.join(RESULT_DIR, "top10_images")

TOP_K = 10

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_IMAGE_BASE_DIR, exist_ok=True)


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


def get_query_images(query_dir):
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(query_dir, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(query_dir, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(query_dir, "*.png")))
    return image_paths


def search_one_image(query_image_path, db_items):
    query_name = os.path.splitext(os.path.basename(query_image_path))[0]

    print(f"\n===== 검색 시작: {query_image_path} =====")

    query_ocr_text = azure_ocr(query_image_path)
    query_image_vector = azure_image_vector(query_image_path)
    query_text_vector = azure_text_vector(query_ocr_text) if query_ocr_text else None

    query_has_image = query_image_vector is not None
    query_has_text = query_text_vector is not None

    similarity_results = []

    for item in db_items:
        db_image_vector = item.get("image_vector")
        db_text_vector = item.get("text_vector")

        db_has_image = db_image_vector is not None
        db_has_text = db_text_vector is not None

        image_similarity = None
        text_similarity = None

        if query_has_image and db_has_image:
            image_similarity = cosine_similarity(query_image_vector, db_image_vector)

        if query_has_text and db_has_text:
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

        similarity_results.append({
            "fileName": item.get("fileName"),
            "img_id": item.get("img_id"),
            "ocr_text": item.get("ocr_text", ""),
            "type": result_type,
            "image_similarity": image_similarity,
            "text_similarity": text_similarity,
            "final_score": final_score,
            "niceCode": item.get("niceCode", []),
            "tot_mid_vienna_code": item.get("tot_mid_vienna_code", []),
            "tot_vienna_code": item.get("tot_vienna_code", []),
            "applicant": item.get("applicant"),
            "classOfGoodServiceBusinessName": item.get("classOfGoodServiceBusinessName"),
            "application_date": item.get("application_date")
        })

    similarity_results.sort(key=lambda x: x["final_score"], reverse=True)
    top10 = similarity_results[:TOP_K]

    for idx, item in enumerate(top10, start=1):
        item["rank"] = idx

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
            print(f"이미지 없음: {file_name}")
            missing_count += 1

    result_json_path = os.path.join(RESULT_DIR, f"{query_name}_top10_result.json")

    output = {
        "query": {
            "query_image": query_image_path,
            "query_ocr_text": query_ocr_text,
            "query_type": get_type(query_has_image, query_has_text)
        },
        "total_candidates": len(similarity_results),
        "top_k": TOP_K,
        "results": top10
    }

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"JSON 저장: {result_json_path}")
    print(f"Top10 이미지 저장: {result_image_dir}")
    print(f"이미지 복사 성공: {copied_count}")
    print(f"이미지 없음: {missing_count}")


def main():
    check_env()

    with open(VECTOR_JSON_PATH, "r", encoding="utf-8") as f:
        db_items = json.load(f)

    query_images = get_query_images(QUERY_IMAGE_DIR)

    if not query_images:
        print(f"검색할 이미지가 없습니다: {QUERY_IMAGE_DIR}")
        return

    print(f"검색 이미지 개수: {len(query_images)}")

    for query_image_path in query_images:
        search_one_image(query_image_path, db_items)

    print("\n===== 전체 완료 =====")


if __name__ == "__main__":
    main()