import os
import json
import shutil

JSON_PATH = "2022_TOTAL.json"
IMAGE_DIR = "image/2022"

OUTPUT_IMAGE_DIR = "test"
OUTPUT_JSON_PATH = "test/test.json"

# 원하는 niceCode들
TARGET_CODES = ["020"]

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# 실제 파일명 매핑 (대소문자 대응)
real_files = {
    f.lower(): f
    for f in os.listdir(IMAGE_DIR)
}

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_images = []

copied_count = 0
missing_count = 0

for item in data["images"]:

    nice_codes = item.get("niceCode", [])

    # 하나라도 TARGET_CODES에 포함되면 통과
    if any(code in TARGET_CODES for code in nice_codes):

        filename = item["fileName"]

        real_name = real_files.get(filename.lower())

        if real_name:

            src_path = os.path.join(IMAGE_DIR, real_name)
            dst_path = os.path.join(OUTPUT_IMAGE_DIR, real_name)

            shutil.copy2(src_path, dst_path)

            filtered_images.append(item)

            copied_count += 1

        else:
            print(f"이미지 없음: {filename}")
            missing_count += 1

# JSON 저장
output_data = {
    "images": filtered_images
}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("\n===== 완료 =====")
print(f"대상 niceCode: {TARGET_CODES}")
print(f"JSON 저장 개수: {len(filtered_images)}")
print(f"이미지 복사 성공: {copied_count}")
print(f"이미지 없음: {missing_count}")

print(f"\n이미지 폴더: {OUTPUT_IMAGE_DIR}")
print(f"JSON 파일: {OUTPUT_JSON_PATH}")