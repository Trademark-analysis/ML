import os
import json
import shutil
import numpy as np
from dotenv import load_dotenv

# 기존 코드에서 가져오기
from ocr import azure_ocr, azure_image_vector, azure_text_vector, check_env

load_dotenv()

VECTOR_JSON_PATH = "trademark_vector.json"
DB_IMAGE_DIR = "image_filtered"

QUERY_IMAGE_PATH = "query.jpg"

RESULT_DIR = "result"
RESULT_IMAGE_DIR = os.path.join(RESULT_DIR, "top10_images")
RESULT_JSON_PATH = os.path.join(RESULT_DIR, "top10_result.json")

TOP_K = 10

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)


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


def main():
    check_env()

    with open(VECTOR_JSON_PATH, "r", encoding="utf-8") as f:
        db_items = json.load(f)

    # 1. 신규 이미지 OCR + 벡터 생성
    query_ocr_text = azure_ocr(QUERY_IMAGE_PATH)
    query_image_vector = azure_image_vector(QUERY_IMAGE_PATH)
    query_text_vector = azure_text_vector(query_ocr_text) if query_ocr_text else None

    query_has_image = query_image_vector is not None
    query_has_text = query_text_vector is not None

    similarity_results = []

    # 2. 기존 상표들과 유사도 계산
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

            # 기존 메타데이터도 같이 저장
            "niceCode": item.get("niceCode", []),
            "tot_mid_vienna_code": item.get("tot_mid_vienna_code", []),
            "tot_vienna_code": item.get("tot_vienna_code", []),
            "applicant": item.get("applicant"),
            "classOfGoodServiceBusinessName": item.get("classOfGoodServiceBusinessName"),
            "application_date": item.get("application_date")
        })

    # 3. top10 정렬
    similarity_results.sort(key=lambda x: x["final_score"], reverse=True)
    top10 = similarity_results[:TOP_K]

    for idx, item in enumerate(top10, start=1):
        item["rank"] = idx

    # 4. top10 이미지 저장
    copied_count = 0
    missing_count = 0

    for item in top10:
        file_name = item["fileName"]

        src_path = os.path.join(DB_IMAGE_DIR, file_name)
        dst_path = os.path.join(RESULT_IMAGE_DIR, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"이미지 없음: {file_name}")
            missing_count += 1

    # 5. top10 JSON 저장
    output = {
        "query": {
            "query_image": QUERY_IMAGE_PATH,
            "query_ocr_text": query_ocr_text,
            "query_type": get_type(query_has_image, query_has_text)
        },
        "total_candidates": len(similarity_results),
        "top_k": TOP_K,
        "results": top10
    }

    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n===== 완료 =====")
    print(f"Top10 JSON 저장: {RESULT_JSON_PATH}")
    print(f"Top10 이미지 저장: {RESULT_IMAGE_DIR}")
    print(f"이미지 복사 성공: {copied_count}")
    print(f"이미지 없음: {missing_count}")


if __name__ == "__main__":
    main()