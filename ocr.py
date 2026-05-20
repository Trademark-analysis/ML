import requests
import time
import json
import glob
import os

KEY = os.getenv("AZURE_VISION_KEY")
ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")

analyze_url = ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

headers = {
    "Ocp-Apim-Subscription-Key": KEY,
    "Content-Type": "application/octet-stream"
}

# 폴더 안 전체 jpg
image_files = glob.glob("010109_TOTAL/*.jpg")

results = []

for image_path in image_files:

    print(f"\n처리중: {image_path}")

    with open(image_path, "rb") as f:
        response = requests.post(
            analyze_url,
            headers=headers,
            data=f
        )

    if response.status_code != 202:
        print("OCR 요청 실패")
        continue

    operation_url = response.headers["Operation-Location"]

    while True:

        result = requests.get(
            operation_url,
            headers={
                "Ocp-Apim-Subscription-Key": KEY
            }
        ).json()

        status = result["status"]

        if status == "succeeded":
            break

        if status == "failed":
            print("OCR 분석 실패")
            break

        time.sleep(1)

    texts = []

    if status == "succeeded":

        for page in result["analyzeResult"]["readResults"]:

            for line in page["lines"]:

                texts.append(line["text"])

    ocr_text = " ".join(texts)

    print("OCR 결과:", ocr_text)

    results.append({
        "image_path": image_path,
        "ocr_text": ocr_text
    })

# 저장
with open("ocr_results.json", "w", encoding="utf-8") as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n전체 OCR 완료")