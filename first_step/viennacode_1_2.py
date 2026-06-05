"""
1차 상표 비엔나 코드 분류 파이프라인 - API 반환용 최종 버전
- 입력: {"image": "Blob URL 또는 로컬 이미지 경로", "nice_codes": [...], "similar_group_codes": [...]}
- 처리: Custom Vision + GPT-4o Vision OR 조합으로 비엔나 코드 추출
- 후보: 비엔나 코드 + 입력받은 니스코드/유사군코드 기준 primary/secondary 분리
- 출력: dict(JSON 직렬화 가능한 형태) return
"""
from pathlib import Path
import base64
import json
import os
from typing import Any, BinaryIO, Dict, List
from first_step.nicecode_1_1 import classify_first_step, classify_second_step

import requests
from urllib.parse import urlparse
import shutil
from openai import AzureOpenAI
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.prediction.models import ImageUrl

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

IMAGE_DIR = PROJECT_DIR / "image_filtered"
DB_JSON_PATH = PROJECT_DIR / "filtered_niceCode.json"

RESULT_DIR = BASE_DIR / "result"
VIENNA_RESULT_IMAGE_DIR = RESULT_DIR / "top100_images"
VIENNA_RESULT_JSON_PATH = RESULT_DIR / "top100.json"

TOP_K_VIENNA = 100

# Custom Vision
CV_PREDICTION_KEY = os.getenv("CV_PREDICTION_KEY", "")
CV_ENDPOINT = os.getenv("CV_ENDPOINT", "")
CV_PROJECT_ID = os.getenv("CV_PROJECT_ID", "")
CV_PUBLISH_NAME = os.getenv("CV_PUBLISH_NAME", "Iteration2")
CV_THRESHOLD = float(os.getenv("CV_THRESHOLD", "0.2"))

# Azure OpenAI
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# 기존 상표 후보 데이터 JSON Blob URL
DB_URL = os.getenv(
    "DB_URL",
    "https://smuteam2blob.blob.core.windows.net/vienna-pages/filtered_niceCode.json",
)


# Blob Storage에 올라간 비엔나 코드 참조 이미지 URL
VIENNA_PAGE_URLS = {
    "0101": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_023.jpg"],
    "0103": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_025.jpg"],
    "0105": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_027.jpg"],
    "0201": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_038.jpg"],
    "0301": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_053.jpg"],
    "0307": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_062.jpg"],
    "0313": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_070.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_071.jpg",
    ],
    "0315": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_074.jpg"],
    "0501": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_081.jpg"],
    "0503": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_083.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_084.jpg",
    ],
    "0505": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_086.jpg"],
    "0905": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_127.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_128.jpg",
    ],
    "1803": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_182.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_183.jpg",
    ],
    "2401": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_213.jpg"],
    "2409": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_220.jpg"],
    "2411": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_222.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_223.jpg",
    ],
    "2413": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_224.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_225.jpg",
    ],
    "2415": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_226.jpg"],
    "2417": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_227.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_228.jpg",
    ],
    "2503": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_231.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_232.jpg",
    ],
    "2601": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_238.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_239.jpg",
    ],
    "2602": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_241.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_242.jpg",
    ],
    "2603": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_242.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_243.jpg",
    ],
    "2605": [
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_248.jpg",
        "https://smuteam2blob.blob.core.windows.net/vienna-pages/page_249.jpg",
    ],
    "2607": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_250.jpg"],
    "2611": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_252.jpg"],
    "2703": ["https://smuteam2blob.blob.core.windows.net/vienna-pages/page_258.jpg"],
}

SYSTEM_PROMPT = """당신은 상표 비엔나 코드 분류 전문가입니다.
각 비엔나 코드의 공식 분류 기준 이미지를 참고하여 상표를 분류하세요.

[분류 규칙]
1. 이미지에서 보이는 모든 요소에 해당하는 코드를 복수로 반환하세요.
2. 확신이 30% 이상이면 후보에 포함하세요. (1차 필터 목적)
3. 반드시 아래 JSON 형식으로만 응답하세요.

{
  "detected_codes": ["2601", "0103"],
  "reasoning": {
    "2601": "원형 도형 포함",
    "0103": "태양 모양 요소 포함"
  },
  "confidence": {
    "2601": 0.9,
    "0103": 0.7
  }
}"""


def is_url(value: str) -> bool:
    """문자열이 http/https URL인지 확인"""
    parsed = urlparse(value or "")
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def normalize_list(value: Any) -> List[str]:
    """문자열/숫자/리스트 입력을 문자열 리스트로 정규화"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return [text] if text else []
    return []


def get_first_existing(item: Dict[str, Any], keys: List[str]) -> Any:
    """DB 필드명이 조금 달라도 대응하기 위해 후보 키 중 먼저 존재하는 값 반환"""
    for key in keys:
        if key in item:
            return item.get(key)
    return []

def load_db_from_local(json_path: str = DB_JSON_PATH) -> Dict[str, Any]:
    """로컬 filtered_niceCode.json 로드"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"DB JSON 파일을 찾을 수 없습니다: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# 1. Custom Vision 분류
# ──────────────────────────────────────────────
def classify_with_custom_vision(image: str) -> List[str]:
    """
    Custom Vision으로 비엔나 코드 분류
    - image가 Blob URL이면 classify_image_url 사용
    - image가 로컬 경로이면 classify_image 사용
    """
    try:
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": CV_PREDICTION_KEY})
        predictor = CustomVisionPredictionClient(CV_ENDPOINT, credentials)

        if is_url(image):
            results = predictor.classify_image_url(
                CV_PROJECT_ID,
                CV_PUBLISH_NAME,
                ImageUrl(url=image),
            )
        else:
            with open(image, "rb") as f:
                results = predictor.classify_image(
                    CV_PROJECT_ID,
                    CV_PUBLISH_NAME,
                    f.read(),
                )

        codes = []
        for pred in results.predictions:
            if pred.probability >= CV_THRESHOLD:
                codes.append(pred.tag_name)

        return codes

    except Exception:
        # API 응답 구조를 깨지 않기 위해 실패 시 빈 리스트 반환
        return []


# ──────────────────────────────────────────────
# 2. GPT-4o Vision 분류
# ──────────────────────────────────────────────
def classify_with_gpt4o(image: str) -> Dict[str, Any]:
    """
    GPT-4o Vision으로 비엔나 코드 분류
    - image가 Blob URL이면 URL 그대로 전달
    - image가 로컬 경로이면 base64 data URL로 변환 후 전달
    """
    client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        api_version=OPENAI_API_VERSION,
    )

    reference_content = [
        {
            "type": "text",
            "text": "아래는 각 비엔나 코드의 공식 분류 기준 이미지입니다:\n",
        }
    ]

    for code, urls in VIENNA_PAGE_URLS.items():
        reference_content.append({"type": "text", "text": f"\n【{code}】"})
        for url in urls:
            reference_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"},
                }
            )

    if is_url(image):
        image_url = image
    else:
        ext = os.path.splitext(image)[1].lower()
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        with open(image, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        image_url = f"data:{media_type};base64,{img_b64}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": reference_content},
        {"role": "assistant", "content": "분류 기준 확인했습니다. 분류할 이미지를 보내주세요."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 상표 이미지의 비엔나 코드를 분류해주세요:"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": "high"},
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=messages,
        max_tokens=1000,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        result = {"detected_codes": [], "reasoning": {}, "confidence": {}, "raw": raw}

    result.setdefault("detected_codes", [])
    result.setdefault("reasoning", {})
    result.setdefault("confidence", {})

    return result


# ──────────────────────────────────────────────
# 3. OR 조합
# ──────────────────────────────────────────────
def combine_results(cv_codes: List[str], gpt_result: Dict[str, Any]) -> List[str]:
    """Custom Vision + GPT-4o OR 조합"""
    gpt_codes = set(normalize_list(gpt_result.get("detected_codes", [])))
    cv_codes_set = set(normalize_list(cv_codes))
    return sorted(list(gpt_codes | cv_codes_set))


# ──────────────────────────────────────────────
# 4. DB에서 후보 조회
# ──────────────────────────────────────────────
def get_candidates(
    vienna_codes: List[str],
    query_nice_codes: List[str],
    db: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    비엔나 코드 + 입력받은 니스코드/유사군코드 기준 후보 분리

    primary:
        비엔나 코드가 겹치고,
        니스코드 또는 유사군코드가 겹치는 후보

    secondary:
        비엔나 코드는 겹치지만,
        니스코드/유사군코드는 겹치지 않는 후보
    """
    primary = []
    secondary = []
    seen_ids = set()

    code_set = set(normalize_list(vienna_codes))
    nice_set = set(normalize_list(query_nice_codes))

    for item in db.get("images", []):
        item_id = item.get("img_id") or item.get("id") or item.get("fileName")
        if item_id in seen_ids:
            continue

        item_codes = set(normalize_list(item.get("tot_mid_vienna_code", [])))
        item_nice = set(normalize_list(item.get("niceCode", [])))

        # 비엔나 코드가 하나도 겹치지 않으면 1차 후보에서 제외
        if not (item_codes & code_set):
            continue

        seen_ids.add(item_id)

        matched_vienna_codes = sorted(list(item_codes & code_set))
        matched_nice_codes = sorted(list(item_nice & nice_set))

        candidate = {
            "img_id": item.get("img_id"),
            "fileName": item.get("fileName"),
            "applicant": item.get("applicant", ""),
            "vienna_codes": normalize_list(item.get("tot_mid_vienna_code", [])),
            "nice_codes": normalize_list(item.get("niceCode", [])),
            "matched_vienna_codes": matched_vienna_codes,
            "matched_nice_codes": matched_nice_codes,
        }

        if matched_nice_codes:
            primary.append(candidate)
        else:
            secondary.append(candidate)

    primary.sort(
        key=lambda x: (
            len(x["matched_nice_codes"]),
            len(x["matched_vienna_codes"]),
        ),
        reverse=True,
    )

    secondary.sort(key=lambda x: len(x["matched_vienna_codes"]), reverse=True)

    return {"primary": primary, "secondary": secondary}


# ──────────────────────────────────────────────
# 5. 메인 파이프라인 - API에서 이 함수만 호출
# ──────────────────────────────────────────────
def run_vienna_check(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        input_data 예시:
        {
            "image": "사용자 업로드 로고 이미지 로컬 경로",
            "nice_codes": ["025"],
            "similar_group_codes": ["G4501"]
        }

    Returns:
        JSON 직렬화 가능한 dict
    """
    image = input_data.get("image")
    query_nice_codes = normalize_list(input_data.get("nice_codes", []))
    trademark_name = input_data.get("trademark_name", "")

    if not image:
        return {
            "success": False,
            "message": "image 값이 필요합니다.",
            "data": None,
        }

    try:
        cv_codes = classify_with_custom_vision(image)
        gpt_result = classify_with_gpt4o(image)
        final_codes = combine_results(cv_codes, gpt_result)

        db = load_db_from_local(DB_JSON_PATH)

        candidates = get_candidates(
            vienna_codes=final_codes,
            query_nice_codes=query_nice_codes,
            db=db,
        )

        result = {
            "input": {
                "image": image,
                "nice_codes": query_nice_codes,
                "trademark_name": trademark_name
            },
            "detected_vienna_codes": final_codes,
            "detail": {
                "custom_vision_codes": cv_codes,
                "gpt4o_codes": gpt_result.get("detected_codes", []),
                "gpt4o_reasoning": gpt_result.get("reasoning", {}),
                "gpt4o_confidence": gpt_result.get("confidence", {}),
            },
            "candidates": candidates,
            "summary": {
                "primary_count": len(candidates["primary"]),
                "secondary_count": len(candidates["secondary"]),
            },
        }

        return {
            "success": True,
            "message": "1차 상표 비엔나 코드 분류 및 후보 조회 완료",
            "data": result,
        }

    except Exception as e:
        return {
            "success": False,
            "message": "1차 상표 비엔나 코드 분류 및 후보 조회 실패",
            "error": str(e),
            "data": None,
        }

def run_nice_classification(
    image_path: str,
    service_description: str,
    trademark_name: str,
) -> Dict[str, Any]:
    """
    1단계: 니스분류 후보 추천

    입력:
    - 로컬 이미지 경로
    - 상품/서비스 설명
    """

    if not image_path:
        return {
            "success": False,
            "step": "nice_classification",
            "message": "image_path 값이 필요합니다.",
            "data": None,
        }

    if not os.path.exists(image_path):
        return {
            "success": False,
            "step": "nice_classification",
            "message": f"이미지 파일을 찾을 수 없습니다: {image_path}",
            "data": None,
        }

    if not service_description.strip():
        return {
            "success": False,
            "step": "nice_classification",
            "message": "service_description 값이 필요합니다.",
            "data": None,
        }

    try:
        result = classify_first_step(
            image=image_path,
            service_description=service_description,
        )

        result["trademark_name"] = trademark_name

        return {
            "success": True,
            "step": "nice_classification",
            "message": "니스분류 후보 추천 완료",
            "data": result,
        }

    except Exception as e:
        return {
            "success": False,
            "step": "nice_classification",
            "message": "니스분류 후보 추천 실패",
            "error": str(e),
            "data": None,
        }
    
def run_similar_group_classification(
    image: str,
    service_description: str,
    selected_nice_classes: List[int],
    trademark_name: str,
) -> Dict[str, Any]:
    """
    2단계: 유사군 코드 추천

    입력:
    - 1단계에서 반환된 image Blob URL
    - 1단계에서 사용한 service_description
    - 사용자가 선택한 니스분류 selected_nice_classes

    처리:
    classify_second_step()으로 nice_codes + similar_group_codes 생성

    반환:
    {
        "success": true,
        "step": "similar_group_classification",
        "data": {
            "image": "...",
            "nice_codes": ["041"],
            "similar_group_codes": ["S120903"]
        }
    }
    """

    if not image:
        return {
            "success": False,
            "step": "similar_group_classification",
            "message": "image 값이 필요합니다.",
            "data": None,
        }

    if not service_description.strip():
        return {
            "success": False,
            "step": "similar_group_classification",
            "message": "service_description 값이 필요합니다.",
            "data": None,
        }

    if not selected_nice_classes:
        return {
            "success": False,
            "step": "similar_group_classification",
            "message": "selected_nice_classes 값이 필요합니다.",
            "data": None,
        }

    try:
        result = classify_second_step(
            image=image,
            service_description=service_description,
            selected_nice_classes=selected_nice_classes,
        )

        result["trademark_name"] = trademark_name

        return {
            "success": True,
            "step": "similar_group_classification",
            "message": "유사군 코드 추천 완료",
            "data": result,
        }

    except Exception as e:
        return {
            "success": False,
            "step": "similar_group_classification",
            "message": "유사군 코드 추천 실패",
            "error": str(e),
            "data": None,
        }
    
def parse_nice_class_input(user_input: str) -> List[int]:
    """
    터미널에서 입력받은 니스분류 문자열을 int 리스트로 변환합니다.

    입력 예시:
    - "41"
    - "41,42"
    - "25 41"
    - "25, 41, 42"
    """

    if not user_input.strip():
        return []

    raw_parts = user_input.replace(",", " ").split()

    selected = []

    for part in raw_parts:
        try:
            nice_class = int(part)
        except ValueError:
            continue

        if nice_class in {25, 41, 42} and nice_class not in selected:
            selected.append(nice_class)

    return selected

def save_top_vienna_candidates(
    primary_candidates: List[Dict[str, Any]],
    secondary_candidates: List[Dict[str, Any]],
    test_image_info: Dict[str, Any],
    top_k: int = TOP_K_VIENNA,
) -> Dict[str, Any]:
    """
    비엔나 코드 기준 후보를 최대 top_k개 저장한다.
    primary를 먼저 사용하고, 부족하면 secondary로 채운다.
    """

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    VIENNA_RESULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    selected_candidates = []

    for candidate in primary_candidates:
        selected_candidates.append({
            **candidate,
            "match_group": "primary",
        })

        if len(selected_candidates) >= top_k:
            break

    if len(selected_candidates) < top_k:
        for candidate in secondary_candidates:
            selected_candidates.append({
                **candidate,
                "match_group": "secondary",
            })

            if len(selected_candidates) >= top_k:
                break

    saved_results = []

    for rank, candidate in enumerate(selected_candidates, start=1):
        file_name = candidate.get("fileName")

        source_path = IMAGE_DIR / file_name if file_name else None

        saved_image_path = None
        copy_success = False

        if source_path and source_path.exists():
            saved_name = file_name
            target_path = VIENNA_RESULT_IMAGE_DIR / saved_name
            shutil.copy2(source_path, target_path)

            saved_image_path = str(target_path)
            copy_success = True

        saved_results.append({
            "rank": rank,
            "match_group": candidate.get("match_group"),
            "img_id": candidate.get("img_id"),
            "fileName": file_name,
            "vienna_codes": candidate.get("vienna_codes", []),
            "nice_codes": candidate.get("nice_codes", []),
            "matched_vienna_codes": candidate.get("matched_vienna_codes", []),
            "matched_nice_codes": candidate.get("matched_nice_codes", []),
        })

    output = {
        "test_image": test_image_info,
        "primary_count": len(primary_candidates),
        "secondary_count": len(secondary_candidates),
        "total_candidate_count": len(primary_candidates) + len(secondary_candidates),
        "results": saved_results,
    }

    with open(VIENNA_RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output

def run_first_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    trademark_name = input_data.get("trademark_name", "").strip()

    step1 = run_nice_classification(
        image_path=input_data.get("image_path"),
        service_description=input_data.get("service_description", ""),
        trademark_name=trademark_name
    )

    if not step1.get("success"):
        return {
            "success": False,
            "mode": "all_test",
            "message": "STEP 1 실패",
            "step1": step1,
            "step2": None,
            "step3": None,
        }

    step1_data = step1.get("data") or {}
    candidates = step1_data.get("nice_class_candidates", [])

    print("\n[STEP 1] 니스분류 후보")
    if not candidates:
        print("추천된 니스분류 후보가 없습니다.")
        return {
            "success": False,
            "mode": "all_test",
            "message": "니스분류 후보가 없어 STEP 2를 진행할 수 없습니다.",
            "step1": step1,
            "step2": None,
            "step3": None,
        }

    for idx, candidate in enumerate(candidates, start=1):
        nice_class = candidate.get("nice_class")
        class_name = candidate.get("class_name", "")
        confidence = candidate.get("confidence", "")

        print(f"{idx}. {nice_class}류 - {class_name} ({confidence})")

    selected_nice_classes = input_data.get("selected_nice_classes")

    if selected_nice_classes:
        selected_nice_classes = [
            int(code) for code in selected_nice_classes
        ]
    else:
        print("\n선택할 니스분류를 입력하세요.")
        print("예: 41")
        print("예: 41,42")
        print("예: 25 41")

        selected_input = input("선택한 니스분류: ")
        selected_nice_classes = parse_nice_class_input(selected_input)

    if not selected_nice_classes:
        return {
            "success": False,
            "mode": "all_test",
            "message": "유효한 니스분류가 선택되지 않았습니다. 25, 41, 42 중에서 입력해야 합니다.",
            "step1": step1,
            "step2": None,
            "step3": None,
        }

    print(f"\n선택된 니스분류: {selected_nice_classes}")

    step2 = run_similar_group_classification(
        image=step1_data.get("image"),
        service_description=step1_data.get("service_description", ""),
        selected_nice_classes=selected_nice_classes,
        trademark_name=trademark_name
    )

    if not step2.get("success"):
        return {
            "success": False,
            "mode": "all_test",
            "message": "STEP 2 실패",
            "selected_nice_classes": selected_nice_classes,
            "step1": step1,
            "step2": step2,
            "step3": None,
        }

    step2_data = step2.get("data") or {}
    step2_data["trademark_name"] = trademark_name

    print("\n[STEP 2] 유사군 코드 추천 결과")
    print(json.dumps(step2, ensure_ascii=False, indent=2))

    step3 = run_vienna_check(step2_data)

    step3_data = step3.get("data") or {}
    candidates = step3_data.get("candidates", {})
    primary_candidates = candidates.get("primary", [])
    secondary_candidates = candidates.get("secondary", [])

    test_image_info = {
        "image": step1_data.get("image"),
        "service_description": step1_data.get("service_description"),
        "nice_codes": step2_data.get("nice_codes", []),
        "similar_group_codes": step2_data.get("similar_group_codes", []),
        "vienna_codes": step3_data.get("detected_vienna_codes", []),
        "trademark_name": trademark_name
    }

    top100_save_result = save_top_vienna_candidates(
        primary_candidates=primary_candidates,
        secondary_candidates=secondary_candidates,
        test_image_info=test_image_info,
        top_k=TOP_K_VIENNA,
    )

    return {
        "image": step1_data.get("image"),
        "trademark_name": trademark_name,
        "service_description": step1_data.get("service_description"),
        "nice_codes": step2_data.get("nice_codes", []),
        "similar_group_codes": step2_data.get("similar_group_codes", []),
        "vienna_codes": step3_data.get("detected_vienna_codes"),
        "matched_primary_images": primary_candidates,
        "matched_secondary_images": secondary_candidates,
        "matched_primary_count": len(primary_candidates),
        "matched_secondary_count": len(secondary_candidates),
        "matched_total_count": len(primary_candidates) + len(secondary_candidates),
        "saved_top100": top100_save_result,
    }

if __name__ == "__main__":
    test_image_path = "../test/2022_79241717.jpg" 

    trademark_name = input("상표명을 입력하세요: ")
    service_description = input("상품/서비스 설명을 입력하세요: ")

    with open(test_image_path, "rb") as f:
        result = run_first_pipeline({
            "image_path": test_image_path,
            "trademark_name": trademark_name,
            "service_description": service_description,
        })

