"""
상표 니스분류 + 유사군 코드 추천

1. Azure OpenAI가 25/41/42류 중 니스분류 후보 추천
2. 사용자가 니스분류 선택
3. Azure SQL에서 선택된 니스분류의 유사군 코드 후보 조회
4. Azure OpenAI가 SQL 후보 안에서만 최종 유사군 코드 선택
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import pyodbc
from dotenv import load_dotenv
from openai import AzureOpenAI


# ============================================================
# 1. 환경변수 로드
# ============================================================

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE")
AZURE_SQL_USER = os.getenv("AZURE_SQL_USER")
AZURE_SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD")

ALLOWED_NICE_CLASSES = {25, 41, 42}

NICE_CLASS_NAMES = {
    25: "의류, 신발, 모자",
    41: "교육, 훈련, 연예, 스포츠 및 문화활동",
    42: "과학기술 서비스, 소프트웨어 개발, IT 서비스",
}

# ============================================================
# 2. Azure OpenAI 클라이언트
# ============================================================

def get_openai_client() -> AzureOpenAI:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        raise ValueError("Azure OpenAI 환경변수가 설정되지 않았습니다.")

    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=API_VERSION,
    )


# ============================================================
# 3. Azure SQL 연결
# ============================================================

def get_sql_connection():
    if not all([AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_USER, AZURE_SQL_PASSWORD]):
        raise ValueError("Azure SQL 환경변수가 설정되지 않았습니다.")

    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={AZURE_SQL_SERVER};"
        f"DATABASE={AZURE_SQL_DATABASE};"
        f"UID={AZURE_SQL_USER};"
        f"PWD={AZURE_SQL_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    return pyodbc.connect(connection_string)


# ============================================================
# 4. 유틸리티
# ============================================================

def clean_json_response(text: str) -> str:
    """
    LLM 응답에서 ```json 코드블록이 붙는 경우 제거합니다.
    """
    return text.replace("```json", "").replace("```", "").strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    LLM 응답을 JSON으로 파싱합니다.
    실패 시 raw_response를 포함해 반환합니다.
    """
    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "matched": False,
            "message": "LLM 응답 JSON 파싱에 실패했습니다.",
            "raw_response": cleaned,
            "candidates": [],
        }

STOPWORDS = {
    "서비스",
    "상품",
    "제품",
    "제공",
    "운영",
    "관련",
    "기반",
    "위한",
    "하는",
    "하고",
    "합니다",
    "및",
    "또는",
    "회사",
    "사업",
    "업체",
    "브랜드",
}

IMPORTANT_PHRASES = [
    # 25류
    "의류 판매",
    "의류온라인판매",
    "의류 온라인 판매",
    "스포츠 의류",
    "스포츠의류",
    "운동복",
    "패션 의류",
    "패션의류",
    "신발 판매",
    "신발판매",
    "모자 판매",
    "모자판매",

    # 41류
    "온라인 교육",
    "온라인교육",
    "온라인 강의",
    "온라인강의",
    "영어 교육",
    "영어교육",
    "교육 플랫폼",
    "교육플랫폼",
    "교육 서비스",
    "교육서비스",
    "스포츠 시설",
    "스포츠시설",
    "헬스장 운영",
    "헬스장운영",
    "체육관 운영",
    "체육관운영",
    "구독형 콘텐츠",
    "구독형콘텐츠",
    "콘텐츠 제공",
    "콘텐츠제공",

    # 42류
    "소프트웨어 개발",
    "소프트웨어개발",
    "앱 개발",
    "앱개발",
    "어플 개발",
    "어플개발",
    "웹 개발",
    "웹개발",
    "플랫폼 개발",
    "플랫폼개발",
    "IT 서비스",
    "IT서비스",
    "인공지능 개발",
    "인공지능개발",
    "AI 개발",
    "AI개발",
    "시스템 개발",
    "시스템개발",
    "데이터베이스",
    "SaaS",
]

SYNONYM_MAP = {
    "어플": ["앱", "모바일앱", "소프트웨어"],
    "앱": ["어플", "모바일앱", "소프트웨어"],
    "모바일앱": ["앱", "어플", "소프트웨어"],
    "웹사이트": ["웹", "웹개발", "웹 개발"],
    "플랫폼": ["플랫폼개발", "플랫폼 개발", "소프트웨어"],
    "ai": ["AI", "인공지능"],
    "인공지능": ["AI"],

    "강의": ["교육", "온라인교육", "온라인 교육"],
    "수업": ["교육", "강의"],
    "학원": ["교육"],
    "튜터링": ["교육", "수업"],
    "영어": ["영어교육", "영어 교육", "어학원"],

    "옷": ["의류", "패션"],
    "패션": ["의류"],
    "운동복": ["스포츠의류", "스포츠 의류", "의류"],
    "스포츠웨어": ["스포츠의류", "스포츠 의류", "운동복"],
    "운동화": ["신발"],
    "스니커즈": ["신발", "운동화"],

    "헬스장": ["스포츠시설", "스포츠 시설", "체육관"],
    "체육관": ["스포츠시설", "스포츠 시설"],
}


def normalize_text(text: str) -> str:
    """
    SQL 검색어 생성을 위해 텍스트를 정리합니다.
    - 영문 소문자화
    - 특수문자 제거
    - 여러 공백을 하나로 정리
    """
    text = text.lower()
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_spaces(text: str) -> str:
    """
    '온라인 교육'과 '온라인교육'을 같이 잡기 위한 함수입니다.
    """
    return normalize_text(text).replace(" ", "")


def add_spacing_variants(term: str) -> List[str]:
    """
    검색어의 띄어쓰기 변형을 함께 만듭니다.
    예:
    - '온라인 교육' → ['온라인 교육', '온라인교육']
    - '앱개발' → ['앱개발']
    """
    normalized = normalize_text(term)
    no_space = remove_spaces(term)

    variants = [normalized]

    if no_space and no_space != normalized:
        variants.append(no_space)

    return variants


def extract_search_terms(text: str) -> List[str]:
    """
    사용자 입력에서 SQL LIKE 검색에 사용할 검색어를 추출합니다.

    구성:
    1. 기본 토큰
    2. 중요 복합 표현
    3. 띄어쓰기 제거 버전
    4. 동의어 확장
    """
    normalized = normalize_text(text)
    tokens = normalized.split()

    terms = []

    # 1. 기본 토큰
    for token in tokens:
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        terms.extend(add_spacing_variants(token))

    # 2. 중요 복합 표현
    no_space_text = remove_spaces(text)

    for phrase in IMPORTANT_PHRASES:
        normalized_phrase = normalize_text(phrase)
        no_space_phrase = remove_spaces(phrase)

        if normalized_phrase in normalized or no_space_phrase in no_space_text:
            terms.extend(add_spacing_variants(phrase))

    # 3. 동의어 확장
    expanded_terms = list(terms)

    for term in terms:
        key = normalize_text(term)
        no_space_key = remove_spaces(term)

        if key in SYNONYM_MAP:
            expanded_terms.extend(SYNONYM_MAP[key])

        if no_space_key in SYNONYM_MAP:
            expanded_terms.extend(SYNONYM_MAP[no_space_key])

    # 4. 동의어도 띄어쓰기 변형 추가
    final_terms = []

    for term in expanded_terms:
        final_terms.extend(add_spacing_variants(term))

    # 중복 제거
    return list(dict.fromkeys([term for term in final_terms if term]))

# ============================================================
# 5. 니스분류 추천
# ============================================================

NICE_CLASS_SYSTEM_PROMPT = """
너는 한국 상표 출원에서 상품/서비스의 니스분류 후보를 추천하는 보조 AI야.

중요 규칙:
1. 반드시 25류, 41류, 42류 중에서만 후보를 추천해.
2. 25/41/42류와 직접적으로 관련이 약하면 후보를 비워도 돼.
3. 반드시 JSON 형식으로만 응답해. 다른 설명 문장은 포함하지 마.

분류 기준:
- 25류: 의류, 신발, 모자, 패션 상품, 스포츠 의류, 운동복, 유니폼 등
- 41류: 교육, 훈련, 온라인 강의, 학원, 콘텐츠 제공, 공연, 연예, 스포츠시설, 문화활동 등
- 42류: 소프트웨어 개발, 앱 개발, 웹 개발, 플랫폼 개발, SaaS, IT 서비스, 과학기술 연구, 디자인 및 기술 서비스 등

응답 JSON 형식:
{
  "matched": true,
  "candidates": [
    {
      "nice_class": 41,
      "class_name": "교육, 훈련, 연예, 스포츠 및 문화활동",
      "confidence": "high",
    }
  ]
}

후보가 없을 때:
{
  "matched": false,
  "candidates": []
}
"""

def recommend_nice_classes_by_llm(service_description: str) -> Dict[str, Any]:
    """
    Azure OpenAI를 사용해서 25/41/42류 중 니스분류 후보 추천
    """
    if not service_description.strip():
        return {
            "matched": False,
            "candidates": [],
        }

    client = get_openai_client()

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": NICE_CLASS_SYSTEM_PROMPT},
            {"role": "user", "content": service_description},
        ],
        temperature=0.1,
    )

    result_text = response.choices[0].message.content or ""
    result = safe_json_loads(result_text)

    filtered_candidates = []

    for candidate in result.get("candidates", []):
        nice_class = candidate.get("nice_class")

        if nice_class in ALLOWED_NICE_CLASSES:
            filtered_candidates.append(candidate)

    filtered_candidates = filtered_candidates[:5]  # 프론트 예시에 따라 5개 이하로 제한하지만, 실제로는 3개 이하

    if not filtered_candidates:
        return {
            "matched": False,
            "candidates": [],
        }

    return {
        "matched": True,
        "candidates": filtered_candidates,
    }


# ============================================================
# 6. SQL에서 유사군 코드 후보 조회
# ============================================================

def validate_selected_nice_classes(selected_nice_classes: List[int]) -> List[int]:
    """
    프론트에서 넘어온 selected_nice_classes 검증
    25/41/42 이외의 값 제거
    """

    if not selected_nice_classes:
        raise ValueError("선택된 니스분류가 없습니다.")

    valid_classes = [
        nice_class
        for nice_class in selected_nice_classes
        if nice_class in ALLOWED_NICE_CLASSES
    ]

    if not valid_classes:
        raise ValueError("25/41/42류 중 하나 이상을 선택해야 합니다.")

    return valid_classes


def fetch_similar_group_candidates_from_sql(
        selected_nice_classes: List[int],
        service_description: str,
        max_rows: int = 80
    ) -> List[Dict[str, Any]]:
    """
    선택된 니스분류에 해당하는 유사군 코드 후보를 SQL에서 가져옴
    """
    valid_classes = validate_selected_nice_classes(selected_nice_classes)
    search_terms = extract_search_terms(service_description)

    conn = None
    cursor = None

    try:
        conn = get_sql_connection()
        cursor = conn.cursor()

        # 1차: nice_class + 검색어 조건
        candidates = _fetch_similar_group_candidates_core(
            cursor=cursor,
            valid_classes=valid_classes,
            search_terms=search_terms,
            max_rows=max_rows,
        )

        if candidates:
            return candidates
        
        # 2차 fallback: 검색어로 아무것도 안 잡히면 nice_class만으로 조회
        return _fetch_similar_group_candidates_core(
            cursor=cursor,
            valid_classes=valid_classes,
            search_terms=[],
            max_rows=max_rows,
        )

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def _fetch_similar_group_candidates_core(
    cursor,
    valid_classes: List[int],
    search_terms: List[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    """
    SQL 쿼리 실행
    search_terms가 있으면 LIKE 조건을 추가하고,
    없으면 nice_class 조건만 사용
    """

    class_placeholders = ",".join(["?"] * len(valid_classes))
    params: List[Any] = []
    params.extend(valid_classes)

    keyword_clause = ""

    if search_terms:
        keyword_conditions = []

        for term in search_terms:
            keyword_conditions.append(
                """
                (
                    item_name LIKE ?
                    OR description LIKE ?
                    OR keywords LIKE ?
                )
                """
            )

            like_term = f"%{term}%"
            params.extend([like_term, like_term, like_term])

        keyword_clause = " AND (" + " OR ".join(keyword_conditions) + ")"

    query = f"""
        SELECT TOP ({max_rows})
            id,
            nice_class,
            item_name,
            similar_group_code,
            description,
            keywords
        FROM similar_group_codes
        WHERE nice_class IN ({class_placeholders})
        {keyword_clause}
        ORDER BY nice_class, id
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    candidates = []

    for row in rows:
        candidates.append({
            "id": row.id,
            "nice_class": row.nice_class,
            "item_name": row.item_name,
            "similar_group_code": row.similar_group_code,
            "description": row.description or "",
            "keywords": row.keywords or "",
        })

    return candidates

# ============================================================
# 7. LLM이 SQL 후보 안에서 유사군 코드 최종 선택
# ============================================================

SIMILAR_GROUP_SYSTEM_PROMPT = """
너는 한국 상표 출원의 유사군 코드 후보를 선택하는 보조 AI야.

중요 규칙:
1. 유사군 코드는 반드시 사용자가 제공한 후보 목록 안에서만 선택해야 한다.
2. 후보 목록에 없는 유사군 코드는 절대 생성하지 마.
3. 상품/서비스 설명과 직접 관련이 있는 후보만 선택해.
4. 관련성이 약하면 후보를 선택하지 않아도 된다.
5. 같은 유사군 코드가 여러 번 있으면 가장 적절한 항목만 선택해.
6. 반드시 JSON 형식으로만 응답해. 다른 설명 문장은 포함하지 마.

응답 JSON 형식:
{
  "matched": true,
  "selected_codes": [
    {
      "candidate_id": 1,
      "nice_class": 42,
      "similar_group_code": "S123301",
      "item_name": "컴퓨터 소프트웨어 개발업",
      "confidence": "high",
    }
  ]
}

선택할 후보가 없을 때:
{
  "matched": false,
  "selected_codes": []
}
"""


def format_candidates_for_prompt(candidates: List[Dict[str, Any]]) -> str:
    """
    SQL 후보를 LLM 프롬프트에 넣기 좋은 형태로 정리
    """
    lines = []

    for idx, candidate in enumerate(candidates, start=1):
        line = (
            f"{idx}. "
            f"candidate_id: {candidate['id']} | "
            f"nice_class: {candidate['nice_class']} | "
            f"similar_group_code: {candidate['similar_group_code']} | "
            f"item_name: {candidate['item_name']} | "
            f"description: {candidate.get('description', '')} | "
            f"keywords: {candidate.get('keywords', '')}"
        )
        lines.append(line)

    return "\n".join(lines)


def select_similar_group_codes_by_llm(
    service_description: str,
    sql_candidates: List[Dict[str, Any]],
    max_candidates_for_prompt: int = 80,
) -> Dict[str, Any]:
    """
    SQL에서 가져온 유사군 코드 후보 안에서 LLM이 최종 후보 선택
    """

    if not sql_candidates:
        return {
            "matched": False,
            "selected_codes": [],
        }

    # 프롬프트가 너무 길어지는 것을 방지
    limited_candidates = sql_candidates[:max_candidates_for_prompt]

    candidate_text = format_candidates_for_prompt(limited_candidates)

    user_prompt = f"""
사용자 상품/서비스 설명:
{service_description}

아래는 SQL 데이터베이스에서 조회된 유사군 코드 후보 목록입니다.
반드시 이 후보 목록 안에서만 선택하세요.

후보 목록:
{candidate_text}
"""

    client = get_openai_client()

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SIMILAR_GROUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    result_text = response.choices[0].message.content or ""
    result = safe_json_loads(result_text)

    # LLM이 후보 목록 밖의 candidate_id/code를 만들면 제거
    candidate_by_id = {
        candidate["id"]: candidate
        for candidate in limited_candidates
    }

    valid_selected_codes = []

    for selected in result.get("selected_codes", []):
        candidate_id = selected.get("candidate_id")

        if candidate_id not in candidate_by_id:
            continue

        original_candidate = candidate_by_id[candidate_id]

        # 코드도 원본 후보 기준으로 덮어씀
        # LLM이 코드를 잘못 적어도 SQL 원본 값을 사용하기 위함
        valid_selected_codes.append({
            "candidate_id": original_candidate["id"],
            "nice_class": original_candidate["nice_class"],
            "similar_group_code": original_candidate["similar_group_code"],
            "item_name": original_candidate["item_name"],
            "description": original_candidate.get("description", ""),
            "keywords": original_candidate.get("keywords", ""),
            "confidence": selected.get("confidence", "medium"),
        })

    if not valid_selected_codes:
        return {
            "matched": False,
            "selected_codes": [],
        }

    return {
        "matched": True,
        "selected_codes": valid_selected_codes,
    }


# ============================================================
# 8. 유사군 코드 선정
# ============================================================

def recommend_similar_group_codes_after_user_selection(
    service_description: str,
    selected_nice_classes: List[int],
) -> Dict[str, Any]:    
    if not service_description.strip():
        return {
            "matched": False,
            "message": "상품/서비스 설명이 비어 있습니다.",
            "selected_codes": [],
        }

    try:
        valid_classes = validate_selected_nice_classes(selected_nice_classes)
    except ValueError as e:
        return {
            "matched": False,
            "message": str(e),
            "selected_codes": [],
        }

    sql_candidates = fetch_similar_group_candidates_from_sql(
        selected_nice_classes=valid_classes,
        service_description=service_description,
    )

    if not sql_candidates:
        return {
            "matched": False,
            "message": "선택된 니스분류에 해당하는 SQL 후보 데이터가 없습니다.",
            "selected_codes": [],
        }

    return select_similar_group_codes_by_llm(
        service_description=service_description,
        sql_candidates=sql_candidates,
    )


# ============================================================
# 9. 로컬 테스트용
# ============================================================

def test_step_1_only():
    service_description = "온라인 영어 교육 플랫폼 운영 서비스"

    nice_result = recommend_nice_classes_by_llm(service_description)

    print("\n[STEP 1] 니스분류 후보 추천 결과")
    print(json.dumps(nice_result, ensure_ascii=False, indent=2))


def test_step_2_with_manual_selection():
    service_description = "온라인 영어 교육 플랫폼 운영 서비스"

    # 실제 서비스에서는 프론트에서 사용자가 선택한 값이 들어옴
    selected_nice_classes = [41]

    similar_result = recommend_similar_group_codes_after_user_selection(
        service_description=service_description,
        selected_nice_classes=selected_nice_classes,
    )

    print("\n[STEP 2] 유사군 코드 추천 결과")
    print(json.dumps(similar_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_step_1_only()
    test_step_2_with_manual_selection()